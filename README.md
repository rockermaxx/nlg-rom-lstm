# Read-Only Memory Running Input LSTMs (ROM-RI-LSTM)


Natural Language Programming (CS6370) Project.


Our objective in this project is to find NN architectures that are capable of learning surface realisation, sentence planning and content determination end-to-end from raw data (images, parameters etc..) to complete paragraphs at once.

LSTMs naturally seemed the best choice. 
We came up with a hybrid LSTM design that learnt an _input-conditional_ language where the raw inputs are passed through another network before being fed to the LSTM.

This project contains 5 variants of LSTM designs that we tried out:

1. **Running-Input LSTM (RI-LSTM)**
Our first attempt at obtaining input conditioned language models. Takes the data as input at every stage
2. **Running-Input Language Model LSTM (RILM-LSTM)** 
Integrates LM-LSTM with Running-Input LSTM
3. **Input-Initialized LM-LSTM (IILM-LSTM)**
Improves upon RILM-LSTM by leveraging memory aspect of LSTMs
4. **Read Only Memory RILM-LSTM**
Combines a read-only variant of memory-networks with RILM-LSTM to allow the network to copy-paste the inputs into it's output. In other words, in addition to the _word_ that an LSTM outputs at each step, the ROM-LSTM can select an _input semantic_ instead.
5. **Word2Vec RILM-LSTM **
Combines dense representations of Word2Vec with RILM-LSTM to address scalability and variability issues with the model.

## RI-LSTM!

