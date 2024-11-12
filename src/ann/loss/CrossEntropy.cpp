/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/cppFiles/class.cc to edit this template
 */

/* 
 * File:   CrossEntropy.cpp
 * Author: ltsach
 * 
 * Created on August 25, 2024, 2:47 PM
 */

#include "loss/CrossEntropy.h"
#include "ann/functions.h"

CrossEntropy::CrossEntropy(LossReduction reduction): ILossLayer(reduction){
    
}

CrossEntropy::CrossEntropy(const CrossEntropy& orig):
ILossLayer(orig){
}

CrossEntropy::~CrossEntropy() {
}

double CrossEntropy::forward(xt::xarray<double> X, xt::xarray<double> t){
    //YOUR CODE IS HERE
     this->m_aCached_Ypred = X;
    this->m_aYtarget = t;
    // const double EPSILON = 1e-7;
    // int nsamples = X.shape()[0];

    // xt::xarray<double> logYpred = xt::log(X + EPSILON);
    // xt::xarray<double> lossArray = -t * logYpred;

    // lossArray = xt::sum(lossArray, -1);

    // xt::xarray<double> sum = xt::sum(lossArray);

    // if (this->m_eReduction == REDUCE_MEAN)
    //     return (sum / nsamples)[0];

    // else
    //     return sum[0];
  return cross_entropy(X, t, REDUCE_MEAN);
}
xt::xarray<double> CrossEntropy::backward() {
    //YOUR CODE IS HERE
    const double EPSILON = 1e-7;

    // xt::xarray<double> DY = -this->m_aYtarget / (this->m_aCached_Ypred + EPSILON);
    // DY /= this->m_aCached_Ypred.shape()[0];

    // return DY;
    xt::xarray<double> grad = -(m_aYtarget / (m_aCached_Ypred + EPSILON));
  grad /= m_aCached_Ypred.shape(0);
  return grad;
}