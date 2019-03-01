import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  """
  num_samples=X.shape[0]
  num_classes=shape[1]
  for i in range(num_samples):
    scores=np.dot(X[i],W)
    scores-=np.max(scores)
    scores_exp=np.sum(np.exp(scores))
    loss-=np.log(np.exp(scores[y[i]]/scores_exp)
    for j in range(num_classes):
      p_ij=scores[j]/scores_exp
      if j==y[i]:
        dW[:,j]+=(p_ij-1)*X[i,:].T
      else:
        dW[:,j]+=p_ij*X[i,:].T
  loss/=num_samples
  loss+=reg*np.sum(W*W)
  dW/=num_samples
  dW+=2*reg*W     
  """
  num_samples = X.shape[0]

  num_classes = W.shape[1]

  score_row_i = np.zeros(num_classes)

  for i in range(num_samples):

    score_row_i = X[i].dot(W)

    score_row_i -= np.max(score_row_i)  # prevent numeric instability

    loss -= np.log( np.exp(score_row_i[y[i]]) / np.sum(np.exp(score_row_i)) )

    for j in range(num_classes):

      P_ij = np.exp(score_row_i[j]) / np.sum(np.exp(score_row_i))

      if(j == y[i]):

        dW[:, j] += X[i, :].T * (P_ij - 1)

      else:

        pass

        dW[:, j] += X[i, :].T * P_ij



  loss /= num_samples

  dW /= num_samples



  loss += reg * np.sum(W * W)

  dW += 2 * reg * W
  
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_samples=X.shape[0]
  num_classes=W.shape[1]
  scores=X.dot(W)
  scores-=np.max(scores,axis=1,keepdims=True)
  scores=np.exp(scores)/np.sum(np.exp(scores),axis=1,keepdims=True)
  loss=np.sum(-1*np.log(scores[range(num_samples),y]))
  scores[range(num_samples),y]-=1
  dW=np.dot(X.T,scores)
  loss/=num_samples
  dW/=num_samples
  loss+=reg*np.sum(W*W)
  dW+=2*reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

