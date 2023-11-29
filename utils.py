# Import des Librairies

import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
import re
import numpy as np
from numpy.random import multinomial
from scipy.special import digamma, gamma
from scipy.stats import dirichlet
import gensim.corpora as corpora
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from pprint import pprint
import gensim
import pyLDAvis.gensim
import pickle
import pyLDAvis
import os
from sparsebm import generate_LBM_dataset, ModelSelection, LBM
from sparsebm.utils import reorder_rows, reorder_cols, ARI, CARI
from scipy.sparse import coo_matrix

# Déclaration des constantes

M = A.shape[0]  # number of products
P = A.shape[1]  # number of users
V = len(words)  # vocabulary size
K = 12  # number of topics
Q = 12  # cluster row
L = 14  # cluster column
D = np.ones(A.shape)  # for amazon dataset, D is a matrix of ones

alphas = np.ones(K)


def get_topic_n_word(i, j, n):
    txt = df.loc[(df['UserId'] == j) & (df['ProductId'] == i), 'Text']
    txt = txt.values[0][n]
    id_word = id2word.token2id[txt]
    topic = lda_model.get_term_topics(id_word)
    try:
        max_topic = max(topic, key=lambda x: x[1])
        max_topic = max_topic[0]
    except:
        # if the word is not in the vocabulary, we assign it to a random topic
        max_topic = np.random.randint(0, K)
    return max_topic


# p(Aij = 1 | XiqYjl=1) = PIql^Aij * (1 - PIql)^(1 - Aij)

def p_Aij_XY(Aij, PIql):
    prob = PIql**Aij * (1 - PIql)**(1 - Aij)
    return prob

# p(A | Y, X, PI) = prod_{i, j} p(Aij | Yi, Xj, PI)


def p_A_YX(A, Y, X, PI):
    prob = 1
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            for q in range(Y.shape[1]):
                for l in range(X.shape[1]):
                    prob *= p_Aij_XY(A.iloc[i, j], PI.iloc[q, l]
                                     )**(Y.iloc[i, q] * X.iloc[j, l])
    return prob


def phi_ijdnk(i, j, d, n, k, w, gamma, Y, X):
    first_prod = 1
    second_prod = 1
    for v in range(V):
        first_prod *= beta_kv(k, v, A, w, D, N, w, gamma,
                              Y, X)**w[i, j, d, n, v]
    for q in range(Q):
        for l in range(L):
            second_prod *= np.exp(digamma(gamma[q][l][k]) -
                                  digamma(sum(gamma[q][l])))**(Y[i, q]*X[j, l])
    return first_prod*second_prod


def gamma_qlk(q, l, k, alpha, A, Y, X, D, N, b, w, gamma):
    res = alpha[k]
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            for d in range(D[i][j]):
                for n in range(N[i][j][d]):
                    res += A[i][j]*Y[i, q]*X[j, l] * \
                        phi_ijdnk(i, j, d, n, k, w, gamma, Y, X)
    return res


def beta_kv(k, v, A, W, D, N, w, gamma, Y, X):
    res = 0
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            for d in range(D[i][j]):
                for n in range(N[i][j][d]):
                    res += A[i][j]*W[i][j][d][n][v] * \
                        phi_ijdnk(i, j, d, n, k, w, gamma, Y, X)


def pi_ql(q, l, A, Y, X):  # GOOD
    res = 0
    A = A.reset_index(drop=True)
    A.columns = range(A.shape[1])
    shape_A = A.shape
    A = np.matrix(A)
    for i in range(shape_A[0]):
        for j in range(shape_A[1]):
            res += Y[i, q]*X[j, l]*A[i, j]
    return res


def p_q(q, Y):  # GOOD
    return np.sum(Y[:, q])


def s(l, X):  # GOOD
    return np.sum(X[:, l])


def q_theta(theta, Q, L, alpha, A, Y, X, D, N, b, w, gamma):
    prod = 1
    for q in range(Q):
        for l in range(L):
            gammas = [gamma_qlk(q, l, k, alpha, A, Y, X, D,
                                N, b, w, gamma) for k in range(K)]
            prod *= dirichlet.pdf(theta[q, l], alpha=gammas)
    return prod


def q_z(z, i, j, d, n, b, w, gamma, Y, X):
    phi = [phi_ijdnk(i, j, d, n, k, b, w, gamma, Y, X) for k in range(K)]
    return multinomial.pdf(z[i][j][d][n], n=1, p=phi)


def greedy_serch(Y, X):
    for i in range(A.shape[0]):
        q = np.where(Y[i] == 1)[0][0]
        friends = False
        for i_prime in range(A.shape[0]):

            if Y[i_prime, q] == 1 and i_prime != i:
                friends = True
                break
        if friends:
            Final_Clust = q
            bestGain = 0
            for q_prime in range(Q):
                if q_prime != q:
                    Y[i, q] = 0
                    Y[i, q_prime] = 1
                    gain = compute_gain(Y, X, D, N, A, w, gamma, Q, L, K)
                    if gain > bestGain:
                        bestGain = gain
                        Final_Clust = q_prime
            Y[i, q] = 0
            Y[i, Final_Clust] = 1

    for j in range(A.shape[1]):
        l = X[j].idxmax()
        friends = False
        for j_prime in range(A.shape[1]) and j_prime != j:
            if X[j_prime, l] == 1:
                friends = True
                break
        if friends:
            Final_Clust = l
            bestGain = 0
            for l_prime in range(L):
                if l_prime != l:
                    X[j, l] = 0
                    X[j, l_prime] = 1
                    gain = compute_gain(Y, X, D, N, A, w, gamma, Q, L, K)
                    if gain > bestGain:
                        bestGain = gain
                        Final_Clust = l_prime
            X[j, l] = 0
            X[j, Final_Clust] = 1
    return Y, X


def compute_gain(Y, X, D, N, A, w, gamma, Q, L, K):
    gain1 = 0
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            a = A[i, j]
            subres = 0
            for d in range(D[i][j]):
                for n in range(N[i][j][d]):
                    for q in range(Q):
                        for l in range(L):
                            subsubres = 0
                            for k in range(K):
                                subsubres += phi_ijdnk(i, j, d, n, k, w, gamma, Y, X)*(
                                    digamma(gamma[q][l][k]) - digamma(sum(gamma[q][l])))
                            subres += X[j, l]*subsubres
            gain1 += a*subres
    gain2 = 0
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            a = A[i, j]
            subres = 0
            for d in range(D[i][j]):
                for n in range(N[i][j][d]):
                    for q in range(Q):
                        for l in range(L):
                            subsubres = 0
                            for k in range(K):
                                subsubres += phi_ijdnk(i, j, d, n, k, w, gamma, Y, X)*(
                                    digamma(gamma[q][l][k]) - digamma(sum(gamma[q][l])))
                            subres += Y[i, q]*subsubres
            gain2 += a*subres

    return gain1 + gain2


def lower_bound(A, W, D, N, w, gamma, Y, X, alpha, b):

    res0 = 0
    for i in range(M):
        for j in range(P):
            for d in range(D[i][j]):
                for n in range(N[i][j][d]):
                    for k in range(K):
                        for v in range(V):
                            res0 += W[i][j][d][n][v] * \
                                np.log(beta_kv(k, v, A, W, D, N, w, gamma, Y, X))
                        res0 *= phi_ijdnk(i, j, d, n, k, w, gamma, Y, X)
            res0 *= A[i][j]

    res1 = 0
    for i in range(M):
        for j in range(P):
            for d in range(D[i][j]):
                for n in range(N[i][j][d]):
                    for q in range(Q):
                        for l in range(L):
                            for k in range(K):
                                res1 += phi_ijdnk(i, j, d, n, k, w, gamma, Y, X)*(
                                    digamma(gamma[q][l][k]) - digamma(sum(gamma[q][l])))
                            res1 *= Y[i][q]*X[j][l]
            res1 *= A[i][j]

    res2 = 0
    for q in range(Q):
        for l in range(L):
            res2 += (np.log(gamma(sum(alphas))) - sum(np.log(gamma(alphas))) +
                     sum([(alphas[k]-1)*(digamma(gamma[q][l][k]) - digamma(sum(gamma[q][l])))]))

    res3 = 0
    for i in range(M):
        for j in range(P):
            for d in range(D[i][j]):
                for n in range(N[i][j][d]):
                    for k in range(K):
                        res3 += phi_ijdnk(i, j, d, n, k, w, gamma, Y, X) * \
                            np.log(phi_ijdnk(i, j, d, n, k, w, gamma, Y, X))
            res3 *= A[i][j]

    res4 = 0
    for q in range(Q):
        for l in range(L):
            res4 += (np.log(gamma(sum(gamma[q][l]))) - sum(np.log(gamma(gamma[q][l][k]))) + sum(
                [(gamma[q][l][k]-1)*(digamma(gamma[q][l][k]) - digamma(sum(gamma[q][l])))]))

    return res0 + res1 + res2 - res3 - res4


def beta(A, W, D, N, w, gamma, Y, X):
    beta = np.zeros((K, V))
    for k in range(K):
        for v in range(V):
            beta[k, v] = beta_kv(k, v, A, W, D, N, w, gamma, Y, X)

    return beta


def pi(A, Y, X):
    pi = np.zeros((Q, L))
    for q in range(Q):
        for l in range(L):
            pi[q, l] = pi_ql(q, l, A, Y, X)

    return pi


def rho(A, Y, X):
    rho = np.zeros(Q)
    for q in range(Q):
        rho[q] = p_q(q, Y)

    return rho


def sigma(A, Y, X):
    sigma = np.zeros(L)
    for l in range(L):
        sigma[l] = s(l, X)

    return sigma


def VEM_step(A, W, D, N, w, gamma, Y, X):

    beta = beta(A, W, D, N, w, gamma, Y, X)
    pi = pi(A, Y, X)
    rho = rho(A, Y, X)
    sigma = sigma(A, Y, X)

    return beta, pi, rho, sigma
