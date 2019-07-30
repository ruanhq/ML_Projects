# LSTM-Text_Classification

The project mainly regards applying RNN based deep neural network
to perform text classification and apply upper level classifier on the RNN to further improve performance.

Firstly we compare various models:
1. Simple MLP classifier with GRU as component architecture
2. Simple default LSTM classifier
3. Bi-directional LSTM classifier
4. Jointly train a LSTM-based Autoencoder and a MLP classifier attached at the latent space.
5. Jointly train a LSTM-based VAE(Variational Autoencoder) and a MLP classifier attached at the latent space.

Secondly apply more models on top of the predicted features from the first level models:
Note that potentially autoencoder can serve as a powerful method for pre-training, the latent space is also what we need to pay much attention to as it can help us learn a compressed representation of the model(Also it can serve as a dimensionality reduction methodology).
Then for the second step:
1. Apply GBDT(Gradient Boosting Decision Tree) on the predicted features to perform binary classification.
2. Apply XGBoost and fine-tune the parameters on the latent features.

#. Discussion:
1. For acq response in Reuter text dataset, we achieve the test AUPR 0.906 and test AUROC more than 0.98 which is much better than the previous traditional methodologies(for simple LSTM the test AUPR is around 0.87).
2. For imbalanced data, Accuracy and AUROC is not that informative compared to AUPR as we are particularly interested in those rare instances.


#. Lessons learned:
Autoencoder is very powerful in manifold learning and if the data has labels, we can attach a MLP classifier at the latent space and jointly train them together.
