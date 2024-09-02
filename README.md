Grammar-Checker
This project aims to correct simple grammatical mistakes using deep learning techniques, specifically a sequence-to-sequence model with an attention mechanism.

Dataset
The process begins by generating input-output pairs for training. This is done by:

Drawing a sample sentence from the dataset.
Applying random perturbations to this sentence to introduce grammatical errors.
Setting the output sequence as the original, unaltered sentence.
The perturbations currently implemented include:

Removing articles (e.g., "a," "an," "the").
Replacing common homophones with their counterparts (e.g., replacing "their" with "there," "then" with "than").
Each perturbation is applied to 25% of the sentences where it could be relevant. Additional grammatical perturbations are welcome and can be easily integrated into the project.

Training
With this augmented dataset, the training process follows a method similar to TensorFlow's sequence-to-sequence tutorial. We use LSTM encoders and decoders with an attention mechanism, optimized using Stochastic Gradient Descent (SGD).

Decoding
While a standard decoder could be used, a more effective approach is employed given that the grammatical errors span a fixed subdomain. The decoder is configured to ensure that all tokens in the sequence either exist in the input sample or belong to a set of corrective tokens provided during training.

This is achieved by modifying the seq2seq model's decoding loop and implementing a post-processing step known as biased decoding. Biased decoding restricts the model to selecting tokens from the input sequence or the corrective token set by applying a binary mask to the logits before extracting predictions.

Note that this logic is not used during training to avoid eliminating potentially valuable information for the model.

Handling Out-of-Vocabulary (OOV) Tokens
Even with biased decoding, the model might still output unknown tokens for OOV words. This project handles such cases within the truncated vocabulary used by the model.

Code Structure
This project builds upon TensorFlow's Seq2SeqModel, extending it slightly to include the biased decoding logic. The main contributions are:

data_reader.py: An abstract class that defines the interface for reading a source dataset and generating input-output pairs, where the input is a grammatically incorrect version of a sentence, and the output is the original sentence.
text_corrector_data_readers.py: Implements several DataReader classes using the Cornell Movie-Dialogs Corpus.
text_corrector_models.py: Contains a modified version of Seq2SeqModel that implements biased decoding.
correct_text.py: A set of helper functions for training the model and decoding sequences with grammatical errors.
TextCorrector.ipynb: An IPython notebook that integrates all components to preprocess text and train the model.
Example Usage
This project is compatible with TensorFlow >= 1.2 and Python 3.5/3.6.

Preprocess Movie Dialog Data
bash
Copier le code
python preprocessors/preprocess_movie_dialogs.py --raw_data movie_lines.txt \
                                                 --out_file preprocessed_movie_lines.txt
The preprocessed file can then be divided into training, validation, and testing sets as needed.

Training
bash
Copier le code
python correct_text.py --train_path /movie_dialog_train.txt \
                       --val_path /movie_dialog_val.txt \
                       --config DefaultMovieDialogConfig \
                       --data_reader_type MovieDialogReader \
                       --model_path /movie_dialog_model
Testing
bash
Copier le code
python correct_text.py --test_path /movie_dialog_test.txt \
                       --config DefaultMovieDialogConfig \
                       --data_reader_type MovieDialogReader \
                       --model_path /movie_dialog_model \
                       --decode
