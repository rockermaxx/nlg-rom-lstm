# TODO:
# 1. Fix broken functions
# 2. Network architecture changed, but functions need to use mask differently
# 3. Evaluation function

from __future__ import print_function

import cPickle as pickle
import sys
import time
from collections import OrderedDict

import numpy
import theano
import theano.tensor as tensor
from theano import config
#from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.tensor.shared_randomstreams import RandomStreams

import encoder_decoder

# datasets = {'imdb': (imdb.load_data, imdb.prepare_data)}

# Set the random number generators' seeds for consistency
SEED = 123
numpy.random.seed(SEED)
trng2 = RandomStreams(1234)


def numpy_floatX(data):
    return numpy.asarray(data, dtype=config.floatX)


def get_minibatches_idx(n, minibatch_size, shuffle=False):
    """
    Used to shuffle the dataset at each iteration.
    """

    idx_list = numpy.arange(n, dtype="int32")

    if shuffle:
        numpy.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
        minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if (minibatch_start != n):
        # Make a minibatch out of what is left
        minibatches.append(idx_list[minibatch_start:])

    return zip(range(len(minibatches)), minibatches)


def zipp(params, tparams):
    """
    When we reload the model. Needed for the GPU stuff.
    """
    for kk, vv in params.items():
        tparams[kk].set_value(vv)


def unzip(zipped):
    """
    When we pickle the model. Needed for the GPU stuff.
    """
    new_params = OrderedDict()
    for kk, vv in zipped.items():
        new_params[kk] = vv.get_value()
    return new_params


def dropout_layer(state_before, use_noise, trng):
    proj = tensor.switch(use_noise,
                         (state_before *
                          trng.binomial(state_before.shape,
                                        p=0.5, n=1,
                                        dtype=state_before.dtype)),
                         state_before * 0.5)
    return proj


def _p(pp, name):
    return '%s_%s' % (pp, name)


def init_params(options):
    """
    Global (not LSTM) parameter. For the embedding and the classifier.
    """
    params = OrderedDict()

    # TODO(biteandbytes) : params['Wemb'] is not required anymore. Remove ?
    # embedding
    # randn = numpy.random.rand(options['n_words'],
    #                          options['dim_proj'])
    # params['Wemb'] = (0.01 * randn).astype(config.floatX)
    params = get_layer(options['encoder'])[0](options,
                                              params,
                                              prefix=options['encoder'])
    # classifier
    params['U'] = 0.01 * numpy.random.randn(options['memdim'] + 1, options['dim_proj']).astype(config.floatX)
    params['b'] = numpy.zeros((options['memdim'] + 1,)).astype(config.floatX)

    params['U_context'] = 0.01 * numpy.random.randn(options['ydim'], options['dim_proj']).astype(config.floatX)
    params['b_context'] = numpy.zeros((options['ydim'],)).astype(config.floatX)

    return params


def load_params(path, params):
    pp = numpy.load(path)
    for kk, vv in params.items():
        if kk not in pp:
            raise Warning('%s is not in the archive' % kk)
        params[kk] = pp[kk]

    return params


def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.items():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams


def get_layer(name):
    fns = layers[name]
    return fns


def ortho_weight(ndim):
    W = numpy.random.randn(ndim, ndim)
    u, s, v = numpy.linalg.svd(W)
    return u.astype(config.floatX)


def param_init_lstm(options, params, prefix='lstm'):
    """
    Init the LSTM parameter:

    :see: init_params
    """
    # Weight matrices
    # Hidden state of dims |H|
    # Input state of dims 3
    # Cell state of dims |H|
    # 4 matrices of size |H|*|X+W+H|
    # TODO: Better if orthogonal?
    weight = 0.01 * numpy.random.randn( 4, options['dim_proj'], options['x_size'] + options['dim_proj'] );

    # Bias vectors of length |H|
    # 4 for each of the above matrices
    bias = numpy.zeros(( 4, options['dim_proj'] ))


    params['weight'] = weight
    params['bias'] = bias.astype(config.floatX)

    return params

def lstm_spass(tparams, state_below, options, prefix='lstm', mask=None, trng=None, memory=None):
    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    assert mask is not None
    assert memory is not None
    #assert trng is not None
    # NOT USED, REMOVE ?
    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]


    # Dims
    # m_ : N
    # W  : Hx(X+W+H)
    # B  : H
    # x_ : Nx(X+W)
    # h_ : NxH
    # c_ : NxH
    # w_ : NxW
    # NOTE(bitesandbytes): WHY THE CHANGE IN CONVENTION? Always keep N and T on top. Becomes extremely confusing especially when the rest
    # of the code is N major.
    # TODO(bitesandbytes) Use _p( prefix, "weight" ) other wise we can't stack LSTMs properly.
    def _step_2(m_, x_, r_, h_, c_, w_, y_):
        # Concat x_, h_ and w_ to get Nx(X+W+H) matrix
        ip_mat = tensor.concatenate([x_, w_, h_], axis=1 )

        # Compute forget gate values
        # f : NxH matrix
        f = tensor.nnet.sigmoid(
            tensor.tensordot(ip_mat, tparams['weight'][0], axes=[1, 1]) + tparams['bias'][0, :][None, :])
        #f = tensor.nnet.sigmoid(tensor.dot(tparams['weight'][0, :, :], ip_mat) + tparams['bias'][0, :][:, None])

        # Compute input gate values
        # i : NxH matrix
        i = tensor.nnet.sigmoid(tensor.tensordot(ip_mat, tparams['weight'][1], axes=[1,1]) + tparams['bias'][1, :][None, :])
        #i = tensor.nnet.sigmoid(tensor.dot(tparams['weight'][1, :, :], ip_mat) + tparams['bias'][1, :][:, None])

        #c_new : NxH matrix
        c_new = tensor.tanh(tensor.tensordot(ip_mat, tparams['weight'][2], axes=[1,1]) + tparams['bias'][2, :][None, :])
        #c_new = tensor.tanh(tensor.dot(tparams['weight'][2, :, :], ip_mat) + tparams['bias'][2, :][:, None])

        # Compute new memory
        # c : NxH
        c = i * c_new + f * c_
        # Retain based on mask
        c = m_[:, None] * c + (1. - m_)[:, None] * c_

        # Compute new hidden state
        # h : NxH
        h = tensor.nnet.sigmoid(
            tensor.tensordot(ip_mat, tparams['weight'][3], axes=[1,1]) + tparams['bias'][3, :][None, :]) * tensor.tanh(c)
        #h = tensor.nnet.sigmoid(
        #    tensor.dot(tparams['weight'][3, :, :], ip_mat) + tparams['bias'][3, :][:, None]) * tensor.tanh(c)
        # Retain based on mask
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        # Predict next vector here.
        # U = OxH.
        # B = O.
        context = tensor.tensordot( h, tparams['U'], axes=[1,1] ) + tparams['b'][None, :]

        y_old = tensor.tensordot( h, tparams['U_context'], axes=[1,1] ) + tparams['b_context'][None, :]
        #y_old = tensor.nnet.softmax(y_old)

        # pred = NxO
        #pred = tensor.nnet.softmax( proj );
        # Nx(M+1)
        #context = tensor.nnet.softmax(context)

        #temp: NxW
        y = tensor.nnet.softmax( ( tensor.sum(context[:, :-1, None ] * memory, axis=1) + context[:, -1][:, None] * y_old ) / options['sample_temperature'] )
        #temp = tensor.sum(temp)

        # ArgMax?
        # pred[ T.arange(pred.shape[0])[:,None], T.arange(pred.shape[1])[None,:], pred.argmax( axis=2 ) ] = 1.;
        # Or Sample from last axis?
        # TxNxO Last dimension one-hot sampled.
        #w = trng2.multinomial( pvals=pred );

        # N
        w_nums = ( tensor.switch( tensor.gt( r_, tensor.extra_ops.cumsum( y, axis=1 ) ), 1, 0 ) ).sum( axis=1 );
        #pred[ tensor.arange(pred.shape[0])[:,None], tensor.arange(pred.shape[1])[None,:], w_nums ] = 1.;
        # NxW
        w = tensor.extra_ops.to_one_hot( w_nums, options['ydim'], dtype=config.floatX)
        return h, c, w.astype(config.floatX), y


    # No idea why this is here. :/
    # TODO(saipraveenb, akshay-balaji, rockermaxx) : Remove this ?
    # NOTE: These are the inputs X. It is called state_below to allow for stacking multiple LSTMs on top of each other.
    # And yes.. not needed anymore. This was the original Wx computation.
    #state_below = (tensor.dot(state_below, tparams[_p(prefix, 'W')]) +
    #               tparams[_p(prefix, 'b')])

    #state_below = trng.multinomial(pvals=state_below);
    # TODO(saipraveenb) : Can you fix this scan function ?
    dim_proj = options['dim_proj']
    word_size = options['ydim']
    rands = trng.uniform( (nsteps,state_below.shape[1],word_size) );
    w_0 = tensor.alloc(numpy_floatX(0.),n_samples, word_size-1);
    w_0_part = tensor.alloc( numpy_floatX(1.), n_samples, 1 );
    w_0 = tensor.concatenate( [w_0_part, w_0], axis=1 );
    rval, updates = theano.scan(_step_2,
                                sequences=[mask, state_below, rands],
                                outputs_info=[tensor.alloc(numpy_floatX(0.), n_samples, dim_proj),
                                              tensor.alloc(numpy_floatX(0.), n_samples, dim_proj),
                                              w_0,
                                              tensor.alloc(numpy_floatX(0.), n_samples, word_size),
                                              ],
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps)
    # Return the words.
    return rval[2]


def lstm_layer(tparams, state_below, options, prefix='lstm', mask=None, memory=None):
    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    assert mask is not None
    assert memory is not None

    # NOT USED, REMOVE ?
    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]

    # Dims
    # m_ : N
    # W  : Hx(X+W+H)
    # B  : H
    # x_w_ : Nx(X+W)
    # h_ : NxH
    # c_ : NxH
    # NOTE(bitesandbytes): WHY THE CHANGE IN CONVENTION? Always keep N and T on top. Becomes extremely confusing especially when the rest
    # of the code is N major.
    # TODO(bitesandbytes) Use _p( prefix, "weight" ) other wise we can't stack LSTMs properly.
    def _step(m_, x_w_, h_, c_, y_):
        # Concat x_w_, h_ to get Nx(X+W+H) matrix
        ip_mat = tensor.concatenate([x_w_, h_], axis=1 )

        # Compute forget gate values
        # f : NxH matrix
        f = tensor.nnet.sigmoid(
            tensor.tensordot(ip_mat, tparams['weight'][0], axes=[1, 1]) + tparams['bias'][0, :][None, :])
        #f = tensor.nnet.sigmoid(tensor.dot(tparams['weight'][0, :, :], ip_mat) + tparams['bias'][0, :][:, None])

        # Compute input gate values
        # i : NxH matrix
        i = tensor.nnet.sigmoid(tensor.tensordot(ip_mat, tparams['weight'][1], axes=[1,1]) + tparams['bias'][1, :][None, :])
        #i = tensor.nnet.sigmoid(tensor.dot(tparams['weight'][1, :, :], ip_mat) + tparams['bias'][1, :][:, None])

        #c_new : NxH matrix
        c_new = tensor.tanh(tensor.tensordot(ip_mat, tparams['weight'][2], axes=[1,1]) + tparams['bias'][2, :][None, :])
        #c_new = tensor.tanh(tensor.dot(tparams['weight'][2, :, :], ip_mat) + tparams['bias'][2, :][:, None])

        # Compute new memory
        # c : NxH
        c = i * c_new + f * c_
        # Retain based on mask
        c = m_[:, None] * c + (1. - m_)[:, None] * c_
        # Compute new hidden state
        # h : NxH
        h = tensor.nnet.sigmoid(
            tensor.tensordot(ip_mat, tparams['weight'][3], axes=[1,1]) + tparams['bias'][3, :][None, :]) * tensor.tanh(c)
        #h = tensor.nnet.sigmoid(
        #    tensor.dot(tparams['weight'][3, :, :], ip_mat) + tparams['bias'][3, :][:, None]) * tensor.tanh(c)
        # Retain based on mask
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        context = tensor.tensordot(h, tparams['U'], axes=[1, 1]) + tparams['b'][None, :]

        y_old = tensor.tensordot(h, tparams['U_context'], axes=[1, 1]) + tparams['b_context'][None, :]
        #y_old = tensor.nnet.softmax(y_old)

        # pred = NxO
        # pred = tensor.nnet.softmax( proj );
        # Nx(M+1)
        #context = tensor.nnet.softmax(context)

        # temp: NxW
        y = tensor.nnet.softmax( tensor.sum(context[:, :-1, None ] * memory, axis=1) + context[:, -1][:, None] * y_old )
        # temp = tensor.sum(temp)

        return h, c, y

    # No idea why this is here. :/
    # TODO(saipraveenb, akshay-balaji, rockermaxx) : Remove this ?
    # NOTE: These are the inputs X. It is called state_below to allow for stacking multiple LSTMs on top of each other.
    # And yes.. not needed anymore. This was the original Wx computation.
    #state_below = (tensor.dot(state_below, tparams[_p(prefix, 'W')]) +
    #               tparams[_p(prefix, 'b')])

    # TODO(saipraveenb) : Can you fix this scan function ?
    dim_proj = options['dim_proj']
    word_size = options['ydim']
    rval, updates = theano.scan(_step,
                                sequences=[mask, state_below],
                                outputs_info=[tensor.alloc(numpy_floatX(0.), n_samples, dim_proj),
                                              tensor.alloc(numpy_floatX(0.), n_samples, dim_proj),
                                              tensor.alloc(numpy_floatX(0.), n_samples, word_size)],
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps)
    return rval[2]
# ff: Feed Forward (normal neural net), only useful to put after lstm
#     before the classifier.
layers = {'lstm': (param_init_lstm, lstm_layer, lstm_spass)}


def sgd(lr, tparams, grads, x, mask, y, cost):
    """ Stochastic Gradient Descent

    :note: A more complicated version of sgd then needed.  This is
        done like that for adadelta and rmsprop.

    """
    # New set of shared variable that will contain the gradient
    # for a mini-batch.
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad' % k)
               for k, p in tparams.items()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    # Function that computes gradients for a mini-batch, but do not
    # updates the weights.
    f_grad_shared = theano.function([x, mask, y], cost, updates=gsup,
                                    name='sgd_f_grad_shared')

    pup = [(p, p - lr * g) for p, g in zip(tparams.values(), gshared)]

    # Function that updates the weights from the previously computed
    # gradient.
    f_update = theano.function([lr], [], updates=pup,
                               name='sgd_f_update')

    return f_grad_shared, f_update


def adadelta(lr, tparams, grads, x, mask, y, memory, cost):
    """
    An adaptive learning rate optimizer

    Parameters
    ----------
    lr : Theano SharedVariable
        Initial learning rate
    tpramas: Theano SharedVariable
        Model parameters
    grads: Theano variable
        Gradients of cost w.r.t to parameres
    x: Theano variable
        Model inputs
    mask: Theano variable
        Sequence mask
    y: Theano variable
        Targets
    cost: Theano variable
        Objective fucntion to minimize

    Notes
    -----
    For more information, see [ADADELTA]_.

    .. [ADADELTA] Matthew D. Zeiler, *ADADELTA: An Adaptive Learning
       Rate Method*, arXiv:1212.5701.
    """

    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.items()]
    running_up2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                 name='%s_rup2' % k)
                   for k, p in tparams.items()]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.items()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function([x, mask, y, memory], cost, updates=zgup + rg2up,
                                    name='adadelta_f_grad_shared')

    updir = [-tensor.sqrt(ru2 + 1e-6) / tensor.sqrt(rg2 + 1e-6) * zg
             for zg, ru2, rg2 in zip(zipped_grads,
                                     running_up2,
                                     running_grads2)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
             for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(tparams.values(), updir)]

    f_update = theano.function([lr], [], updates=ru2up + param_up,
                               on_unused_input='ignore',
                               name='adadelta_f_update')

    return f_grad_shared, f_update


def rmsprop(lr, tparams, grads, x, mask, y, cost):
    """
    A variant of  SGD that scales the step size by running average of the
    recent step norms.

    Parameters
    ----------
    lr : Theano SharedVariable
        Initial learning rate
    tpramas: Theano SharedVariable
        Model parameters
    grads: Theano variable
        Gradients of cost w.r.t to parameres
    x: Theano variable
        Model inputs
    mask: Theano variable
        Sequence mask
    y: Theano variable
        Targets
    cost: Theano variable
        Objective fucntion to minimize

    Notes
    -----
    For more information, see [Hint2014]_.

    .. [Hint2014] Geoff Hinton, *Neural Networks for Machine Learning*,
       lecture 6a,
       http://cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
    """

    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.items()]
    running_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                   name='%s_rgrad' % k)
                     for k, p in tparams.items()]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.items()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rgup = [(rg, 0.95 * rg + 0.05 * g) for rg, g in zip(running_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function([x, mask, y], cost,
                                    updates=zgup + rgup + rg2up,
                                    name='rmsprop_f_grad_shared')

    updir = [theano.shared(p.get_value() * numpy_floatX(0.),
                           name='%s_updir' % k)
             for k, p in tparams.items()]
    updir_new = [(ud, 0.9 * ud - 1e-4 * zg / tensor.sqrt(rg2 - rg ** 2 + 1e-4))
                 for ud, zg, rg, rg2 in zip(updir, zipped_grads, running_grads,
                                            running_grads2)]
    param_up = [(p, p + udn[1])
                for p, udn in zip(tparams.values(), updir_new)]
    f_update = theano.function([lr], [], updates=updir_new + param_up,
                               on_unused_input='ignore',
                               name='rmsprop_f_update')

    return f_grad_shared, f_update


def build_model(tparams, options):
    trng = RandomStreams(SEED)

    # Used for dropout.
    use_noise = theano.shared(numpy_floatX(0.))

    # TxNx(X+W) float( for training ) OR TxNxX float( for prediction )
    x = tensor.tensor3('x', dtype=config.floatX)
    # TxN float( logically boolean )
    mask = tensor.matrix('mask', dtype=config.floatX)
    # TxN int64
    y = tensor.matrix('y', dtype='int64')
    # NxMxW float( Read Only Memory )
    memory = tensor.tensor3('memory', dtype=config.floatX)

    n_timesteps = x.shape[0]
    n_samples = x.shape[1]

    # TODO(biteandbytes) : This gets inputs. Change to n_timesteps*n_samples*3 tensor
    # emb = tparams['Wemb'][x.flatten()].reshape([n_timesteps,
    #                                            n_samples,
    #                                            options['dim_proj']])
    #

    # TxNxH.
    pred = get_layer(options['encoder'])[1](tparams, x, options,
                                            prefix=options['encoder'],

                                            mask=mask,memory=memory)

    # TxNx(X+W) float( for training ) OR TxNxX float( for prediction )
    x2 = tensor.tensor3('x_2', dtype=config.floatX)
    # TxN float( logically boolean )
    mask2 = tensor.matrix('mask_2', dtype=config.floatX)
    # Stochastic forward pass.
    spass = get_layer(options['encoder'])[2](tparams, x2, options,
                                            prefix=options['encoder'],
                                            mask=mask2,
                                            trng=trng,memory=memory)




    #w = trng.multinomial( pvals=w );
    # TODO(biteandbytes) : Modify this ?
    # tparams[U] = HxO # WRONG. It's _'OxH'_
    # O = output one hot vector.
    # H = Hidden state size.
    # NOTE: IT'S 'OxH' NOT 'HxO'. DON'T MENTION THINGS YOU ARE NOT
    # SURE ABOUT! I JUST WASTED AN HOUR.
    """if options['encoder'] == 'lstm':
        #proj = (proj * mask[:, :, None]).sum(axis=0)
        #proj = proj / mask.sum(axis=0)[:, None]
        # B = O

        # TxNxO.
        #proj = tensor.tensordot( proj, tparams['U'], axes=[2,1] ) + tparams['b'][None, None, :];
    """
    # TODO(saipraveenb): Check if we need dropout option.
    #if options['use_dropout']:
    #    proj = dropout_layer(proj, use_noise, trng)

    # pred = TxNxO
    #pred = tensor.nnet.softmax( proj );
    #exp_pred = tensor.exp( proj );
    # TxNxO ( last dimension softmaxed )
    #pred = exp_pred / exp_pred.sum( axis=2, keepdims=True );


    f_pred_prob = theano.function([x, mask, memory], pred, name='f_pred_prob')
    theano.config.exception_verbosity='high'
    #x_2 =
    f_pred = theano.function([x2, mask2, memory], spass.argmax(axis=2), name='f_pred')

    off = 1e-8
    if pred.dtype == 'float16':
        off = 1e-6

    # NOTE: Finished adding the softmax layer with mask.
    cost = -tensor.log( pred[ tensor.arange(pred.shape[0])[:,None], tensor.arange(pred.shape[1])[None,:], y ] * mask + off ).sum()


    return use_noise, x, mask, y, memory, f_pred_prob, f_pred, cost


def pred_probs(f_pred_prob, prepare_data, data, iterator, verbose=False):
    """ If you want to use a trained model, this is useful to compute
    the probabilities of new examples.
    """
    """ DO NOT USE. NOT MAKE ANY SENSE IN THIS CASE """
    n_samples = len(data[0])
    probs = numpy.zeros((n_samples, 2)).astype(config.floatX)

    n_done = 0

    for _, valid_index in iterator:
        x, mask, y = prepare_data([data[0][t] for t in valid_index],
                                  numpy.array(data[1])[valid_index],
                                  maxlen=None, x_dim = 5)
        pred_probs = f_pred_prob(x, mask)
        probs[valid_index, :] = pred_probs

        n_done += len(valid_index)
        if verbose:
            print('%d/%d samples classified' % (n_done, n_samples))

    return probs


def pred_error(f_pred, prepare_data, data, iterator, verbose=False, mapping=None, max_mapping=None, x_dim=5):
    """
    Just compute the error
    f_pred: Theano fct computing the prediction
    prepare_data: usual prepare_data for that dataset.
    """
    valid_err = 0
    for _, valid_index in iterator:
        x, mask, y, memory = prepare_data([data[0][t] for t in valid_index],
                                  numpy.array(data[1])[valid_index],
                                  maxlen=None, x_dim = x_dim, mapping=mapping, max_mapping=max_mapping)
        # TxN
        preds = f_pred(x, mask, memory)
        # TxN
        targets = y;
        #print("TARGET: ")
        #print(targets)
        #print("PRED: ")
        #print(preds);
        valid_err += (preds == targets).sum()
    valid_err = 1. - numpy_floatX(valid_err) / len(data[0])

    return valid_err

# y : TxN int64
# x : TxNx5 float16
def reattach_data( x, y, inpsize=22 ):
    # T-1 x N
    y = y[:-1,:]
    # TxN
    y = numpy.concatenate( (numpy.zeros((1,y.shape[1])),y), axis=0 ).astype(numpy.int16);
    # TxNx22
    x_part = numpy.zeros( y.shape + (inpsize,) )
    x_part[ numpy.arange( y.shape[0] )[:,None], numpy.arange(y.shape[1])[None,:], y ] = 1.;

    # TxNx(X+W)
    return numpy.concatenate( (x, x_part), axis=2 );

def train_lstm(
        dim_proj=128,  # word embeding dimension and LSTM number of hidden units.
        patience=10,  # Number of epoch to wait before early stop if no progress
        max_epochs=10000,  # The maximum number of epoch to run
        dispFreq=10,  # Display to stdout the training progress every N updates
        decay_c=0.,  # Weight decay for the classifier applied to the U weights.
        lrate=0.00005,  # Learning rate for sgd (not used for adadelta and rmsprop)
        n_words=43,  # Vocabulary size
        optimizer=adadelta,
        # sgd, adadelta and rmsprop available, sgd very hard to use, not recommanded (probably need momentum and decaying learning rate).
        encoder='lstm',  # TODO: can be removed must be lstm.
        saveto='lstm_model.npz',  # The best model will be saved there
        validFreq=1000,  # Compute the validation error after this number of update.
        saveFreq=3000,  # Save the parameters after every saveFreq updates
        maxlen=100,  # Sequence longer then this get ignored
        batch_size=64,  # The batch size during training.
        valid_batch_size=16,  # The batch size used for validation/test set.
        dataset='imdb',
        # Parameter for extra option
        noise_std=0.,
        use_dropout=False,  # if False slightly faster, but worst test error
        # This frequently need a bigger model.
        reload_model=None,  # Path to a saved model we want to start from.
        test_size=-1,  # If >0, we keep only this number of test example.
        ydim=176, # Output dimensions.
        w_multiplier=1,
        b_multiplier=1,
        exampleFreq=100,
        inpdim=561,
        sample_temperature=0.1,
        memdim=18
):
    x_size = inpdim + ydim;
    # Model options
    model_options = locals().copy()
    print("model options", model_options)

    print('Loading data')

    # (N*[x], N*[y])
    train, valid, test, vocab = encoder_decoder.get_raw_data("../data/corpus/inputs_x.txt",
                                                      "../data/corpus/targets_x.txt")
    vocab_lst = [''] * ( len(vocab.items()) + 2 );
    for w,i in vocab.items():
        print(i);
        vocab_lst[i] = w;

    # Input - seqs: num_samples*3, labels: num_samples*[list]
    # Return X:maxlen*num_samples*3, X_mask: max_len*num_samples, labels: maxlen*num_samples
    prepare_data = encoder_decoder.prepare_data

    # Chosen as |num words| + 1 (0 -> no word | empty)
    # NOTE: Added ydim as an input to the function and initialized to 22.
    # ydim = 22

    #model_options['ydim'] = ydim

    print('Building model')
    # This create the initial parameters as numpy ndarrays.
    # Dict name (string) -> numpy ndarray
    params = init_params(model_options)

    if reload_model:
        load_params('lstm_model.npz', params)

    # This create Theano Shared Variable from the parameters.
    # Dict name (string) -> Theano Tensor Shared Variable
    # params and tparams have different copy of the weights.
    tparams = init_tparams(params)

    # use_noise is for dropout
    (use_noise, x, mask,
     y, memory, f_pred_prob, f_pred, cost ) = build_model(tparams, model_options)

    if decay_c > 0.:
        decay_c = theano.shared(numpy_floatX(decay_c), name='decay_c')
        weight_decay = 0.
        weight_decay += (tparams['U'] ** 2).sum()
        weight_decay *= decay_c
        cost += weight_decay

    f_cost = theano.function([x, mask, y, memory], cost, name='f_cost')

    grads = tensor.grad(cost, wrt=list(tparams.values()))
    f_grad = theano.function([x, mask, y, memory], grads, name='f_grad')

    lr = tensor.scalar(name='lr')
    f_grad_shared, f_update = optimizer(lr, tparams, grads,
                                        x, mask, y, memory, cost)

    print('Optimization')

    # Random shuffle partition.
    kf_valid = get_minibatches_idx(len(valid[0]), valid_batch_size)
    kf_test = get_minibatches_idx(len(test[0]), valid_batch_size)

    example_test_batch = kf_test[ int( numpy.random.rand() * len(kf_test) ) ][1];
    print("%d train examples" % len(train[0]))
    print("%d valid examples" % len(valid[0]))
    print("%d test examples" % len(test[0]))

    history_errs = []
    best_p = None
    bad_count = 0

    if validFreq == -1:
        validFreq = len(train[0]) # batch_size
    if saveFreq == -1:
        saveFreq = len(train[0]) # batch_size

    uidx = 0  # the number of update done
    estop = False  # early stop
    start_time = time.time()
    try:
        for eidx in range(max_epochs):
            n_samples = 0

            # Get new shuffled index for the training set.
            kf = get_minibatches_idx(len(train[0]), batch_size, shuffle=True)

            for _, train_index in kf:
                uidx += 1
                use_noise.set_value(0.)

                # Select the random examples for this minibatch
                y = [train[1][t] for t in train_index]
                x = [train[0][t] for t in train_index]

                # Get the data in numpy.ndarray format
                # This swap the axis!
                # Return something of shape (minibatch maxlen, n samples)

                # x = TxNx3 float16
                # m = TxN boolean
                # y = TxN int64
                x, mask, y, memory = prepare_data(x, y, x_dim=inpdim, mapping=vocab, max_mapping=ydim)
                #print("MEMORY SIZE:", memory.shape );
                n_samples += x.shape[1]
                # Sample.
                #print("SAMPLE MASK");
                #print( x );
                # x = TxNx(22+5)
                x = reattach_data( x, y, inpsize = ydim );

                cost = f_grad_shared(x, mask, y.astype(numpy.int64), memory)
                f_update(lrate)

                if numpy.isnan(cost) or numpy.isinf(cost):
                    print('bad cost detected: ', cost)
                    return 1., 1., 1.

                if numpy.mod(uidx, dispFreq) == 0:
                    print('Epoch ', eidx, 'Update ', uidx, 'Cost ', cost)

                if numpy.mod(uidx, exampleFreq) == 0:

                    example_index = example_test_batch;
                    x, mask, y, memory = prepare_data([test[0][t] for t in example_index],
                                  numpy.array(test[1])[example_index],
                                  maxlen=None, x_dim = inpdim, mapping=vocab, max_mapping=ydim)

                    # Predict.. don't have to call reattach.
                    # SxNxT
                    preds = numpy.zeros( 10, x.shape[1], x.shape[0] )
                    for i in range(0,10):
                        preds[i] = f_pred(x, mask, memory).transpose().astype(numpy.int64);
                    preds = preds.transpose([1,0,2]);
                    # NxT
                    targets = y.transpose().astype(numpy.int64);

                    k = int( numpy.random.rand() * len(targets) );

                    for a,b,c in zip( x[0], targets, preds ):
                        print( "Targets for x=", a );
                        print( ''.join([ vocab_lst[o] + ' ' for o in b.tolist() ] ) )
                        for p in c:
                            print( "Prediction " );
                            print( ''.join([ vocab_lst[o] + ' ' for o in p.tolist() ] ) )


                if saveto and numpy.mod(uidx, saveFreq) == 0:
                    print('Saving...')

                    if best_p is not None:
                        params = best_p
                    else:
                        params = unzip(tparams)
                    numpy.savez(saveto, history_errs=history_errs, **params)
                    pickle.dump(model_options, open('%s.pkl' % saveto, 'wb'), -1)
                    print('Done')

                if numpy.mod(uidx, validFreq) == 0:
                    use_noise.set_value(0.)
                    train_err = pred_error(f_pred, prepare_data, train, kf, mapping=vocab, max_mapping=ydim, x_dim=inpdim)
                    valid_err = pred_error(f_pred, prepare_data, valid,
                                           kf_valid, mapping=vocab, max_mapping=ydim, x_dim=inpdim)
                    test_err = pred_error(f_pred, prepare_data, test, kf_test, mapping=vocab, max_mapping=ydim, x_dim=inpdim)


                    history_errs.append([valid_err, test_err])

                    #if best_p is None or valid_err <= numpy.array(history_errs)[:, 0].min():
                    #    best_p = unzip(tparams)
                    #    bad_counter = 0

                    print('Train ', train_err, 'Valid ', valid_err,
                          'Test ', test_err)


                    #if len(history_errs) > patience and valid_err >= numpy.array(history_errs)[:-patience, 0].min():
                    #    bad_counter += 1
                    #    if bad_counter > patience:
                    #        print('Early Stop!')
                    #        estop = True
                    #        break

            print('Seen %d samples' % n_samples)

            if estop:
                break

    except KeyboardInterrupt:
        print("Training interupted")

    end_time = time.time()
    if best_p is not None:
        zipp(best_p, tparams)
    else:
        best_p = unzip(tparams)

    use_noise.set_value(0.)
    kf_train_sorted = get_minibatches_idx(len(train[0]), batch_size)
    train_err = pred_error(f_pred, prepare_data, train, kf_train_sorted, mapping=vocab, max_mapping=ydim, x_dim = inpdim)
    valid_err = pred_error(f_pred, prepare_data, valid, kf_valid, mapping=vocab, max_mapping=ydim, x_dim = inpdim)
    test_err = pred_error(f_pred, prepare_data, test, kf_test, mapping=vocab, max_mapping=ydim, x_dim = inpdim)

    print('Train ', train_err, 'Valid ', valid_err, 'Test ', test_err)
    if saveto:
        numpy.savez(saveto, train_err=train_err,
                    valid_err=valid_err, test_err=test_err,
                    history_errs=history_errs, **best_p)
    print('The code run for %d epochs, with %f sec/epochs' % (
        (eidx + 1), (end_time - start_time) / (1. * (eidx + 1))))
    print(('Training took %.1fs' %
           (end_time - start_time)), file=sys.stderr)
    return train_err, valid_err, test_err


if __name__ == '__main__':
    # See function train for all possible parameter and there definition.
    train_lstm(
        max_epochs=1000,
        test_size=500,
        reload_model=True,
    )
