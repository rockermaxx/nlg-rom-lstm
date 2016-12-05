# Learning Input Conditional Language Models for Natural Language Generation

Natural Language Processing (CS6370) Project.


Our objective in this project is to find NN architectures that are capable of learning surface realisation, sentence planning and content determination end-to-end from raw data (images, parameters etc..) to complete paragraphs at once.

LSTMs naturally seemed the best choice. 
We came up with a hybrid LSTM design that learnt an _input-conditional_ language where the raw inputs are passed through another network before being fed to the LSTM.

This project contains 5 variants of LSTM designs that we tried out:

1. **Running-Input LSTM (RI-LSTM)**
Our first attempt at obtaining input conditioned language models. Takes the data as input at every stage
2. **Running-Input Language Model LSTM (RILM-LSTM)** 
Integrates LM-LSTM with Running-Input LSTM
3. **Input-Initialized LM-LSTM (IILM-LSTM)**
Improves upon RILM-LSTM by leveraging long-term memory aspect of LSTMs
4. **Read Only Memory RILM-LSTM**
Combines a read-only variant of memory-networks with RILM-LSTM to allow the network to copy-paste the inputs into it's output. In other words, in addition to the _word_ that an LSTM outputs at each step, the ROM-LSTM can select an _input semantic_ instead.
5. **Word2Vec RILM-LSTM **
Combines dense representations of Word2Vec with RILM-LSTM to address scalability and variability issues with the model.

## RI-LSTM
![](/images/ri_lstm.png)
The set of raw inputs are provided to the LSTM at every stage (the same inputs) along with the previous hidden and cell states. This allows the LSTM to learn an input conditional language model.

## Running Input Language Model LSTM
![]({{site.baseurl}}/images/lm_lstm_train.png)
![]({{site.baseurl}}/images/lm_lstm_test.png)
In the Language Model part, we pass the previous word as an input to the current stage of the LSTM so that it knows which path has been sampled. This is important to allow the LSTM to handle ambuiguous phrases. See the full report for more details.

## Input Initialized LM-LSTM
![]({{site.baseurl}}/images/ii_lstm1.png)
This LSTM initialises the initial cell state with a simple perceptron to transform the input to it's cell state. This part is the input network and can have different structure depending on the nature of inputs. CNNs for images, LSTMs for sequences and standard neural networks for low dimensional parameters.

## Read Only Memory (Copy/Paste) LSTM
![]({{site.baseurl}}/images/rilm_train.png)
![]({{site.baseurl}}/images/rilm_test.png)
This LSTM uses a Memory matrix composed of the word forms of the inputs. The LSTM is optionally allowed to _optionally_ select an _input semantic_ instead of an arbitrary one-hot word output.
This allows for good generalisation on the Prodigy-METEO dataset as the parts of the output text that is simply copy-pasted _input_ are effectively handled.


## Word2Vec LSTM
The Word2Vec LSTM resembles the RILM-LSTM in everything except the output sampling method and the loss function.
- The new loss function is the total vector dot product of the expectations with their corresponding targets.
- The output sampling is performed by using Word2Vec's K similar vectors search based on the expectation and then sampling using the similarity scores.

