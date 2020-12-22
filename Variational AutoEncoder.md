# Variational AutoEncoder

###### tags: `VAE`

> I have made these notes from https://arxiv.org/pdf/1606.05908.pdf. 

- Table of Contents
[ToC]




## Generative Modelling

“Generative modeling” is a broad area of machine learning which deals with models of distributions P(X), defined over datapoints X in some potentially high-dimensional space X . For instance, images are a popular kind of data for which we might create generative models. Each “datapoint” (image) has thousands or millions of dimensions (pixels), and the generative model’s job is to somehow capture the dependencies between pixels, e.g., that nearby pixels have similar color, and are organized into objects. Exactly what it means to “capture” these dependencies depends on exactly what we want to do with the model. One straightforward kind of generative model simply allows us to compute P(X) numerically. In the case of images, X values which look like real images should get high probability, whereas images that look like random noise should get low probability. However, models like this are not necessarily useful: knowing that one image is unlikely does not help us synthesize one that is likely.

Instead, one often cares about producing more examples that are like those already in a database, but not exactly the same. We could start with a database of raw images and synthesize new, unseen images. We might take in a database of 3D models of something like plants and produce more of them to fill a forest in a video game. We could take handwritten text and try to produce more handwritten text. Tools like this might actually be useful for graphic designers. We can formalize this setup by saying that we get examples X distributed according to some unknown distribution Pgt(X), and our goal is to learn a model P which we can sample from, such that P is as similar as possible to Pgt.

:rocket: 

## Prerequisite - Latent Variable Models
- When training a generative model, the more complicated the dependencies between the dimensions, the more difficult the models are to train.
:::info
Example - Generating images of handwritten characters:  

Say for simplicity that we only care about modeling the digits 0-9. If the left half of the character contains the left half of a 5, then the right half cannot contain the left half of a 0, or the character will very clearly not look like any real digit.
:::
- It helps if the model first decides which character to generate before it assigns a value to any specific pixel. This kind of decision is formally called a latent variable.
:::info
It means, before our model draws anything, it first randomly samples a digit value z from the set [0, ..., 9], and then makes sure all the strokes match that character.
:::
- Before we can say that our model is representative of our dataset, we need to make sure that for every datapoint X in the dataset, there is one (or many) settings of the latent variables which causes the model to generate something very similar to X. 
:::info
Let's say we have a vector of latent variables $z$ in a high-dimensional space $Z$ which we can easily sample according to some probability density function (PDF) $P(z)$ defined over $Z$.

Then, say we have a family of deterministic functions $f(z;θ)$, parameterized by a vector $θ$ in some space $Θ$,where $f:Z×Θ→X$.$f$ is deterministic, but if $z$ is random and $θ$ is fixed, then $f(z;θ)$ is a random variable in the space $X$. We wish to optimize $θ$ such that we can sample $z$ from $P(z)$ and, with high probability, $f(z;θ)$ will be like the $X’s$ in our dataset.
:::
- To make this notion precise mathematically, we are aiming maximize the probability of each X in the training set under the entire generative process, according to:
$$
P(X) = \int P(X|z;\theta)P(z)dz
$$
> Here, $f(z;θ)$ has been replaced by a distribution $P(X|z;θ)$, which allows us to make the dependence of $X$ on $z$ explicit by using the law of total probability.

- The intuition behind this framework—called “maximum likelihood”— is that if the model is likely to produce training set samples, then it is also likely to produce similar samples, and unlikely to produce dissimilar ones.

- In VAEs, the choice of this output distribution is often Gaussian, i.e.,
$$
P(X|z;\theta) = \mathcal{N}(X|f(z,\theta),\sigma^{2}*I)
$$
That is,it has mean $f(z;θ)$ and covariance equal to the identity matrix $I$ times some scalar $σ$ (which is a hyperparameter).This replacement is necessary to formalize the intuition that some $z$ needs to result in the samples that are merely like $X$.

- In general, and particularly early in training, our model will not produce outputs that are identical to any particular $X$. By having a Gaussian distribution, we can use gradient descent (or any other optimization technique) to increase $P(X)$ by making $f(z;θ)$ approach $X$ for some $z$, i.e., gradually making the training data more likely under the generative model. 

## Variational Autoencoders 

![](https://i.imgur.com/1QoJ9Ms.png)

VAEs approximately maximize above equation , according to the model shown in Figure 1. They are called “autoencoders” only be- cause the final training objective that derives from this setup does have an encoder and a decoder, and resembles a traditional autoencoder. 

To solve this equation, there are two problems that VAEs must deal with: how to define the latent variables $z$ (i.e., decide what information they represent), and how to deal with the integral over $z$. VAEs give a definite answer to both.

### How to chose latent variables ? 
Returning to our digits example, the ‘latent’ decisions that the model needs to make before it begins painting the digit are actually rather complicated. It needs to choose not just the digit, but the angle that the digit is drawn, the stroke width, and also abstract stylistic properties. Worse, these properties may be correlated: a more angled digit may result if one writes faster, which also might tend to result in a thinner stroke. Ideally, we want to avoid deciding by hand what information each dimension of z encodes. We also want to avoid explicitly describing the dependencies—i.e., the latent structure—between the dimensions of $z$. 

VAEs take an unusual approach to dealing with this problem: they assume that there is no simple interpretation of the dimensions of $z$, and instead assert that samples of $z$ can be drawn from a simple distribution, i.e., $\mathcal{N}(0, I)$, where $I$ is the identity matrix. 

How is this possible ? 
The key is to notice that any distribution in $d$ dimensions can be generated by taking a set of $d$ variables that are normally distributed and mapping them through a sufficiently complicated function1. 

:::info 
![](https://i.imgur.com/Oyn9Yjs.png)
:::

Hence, provided powerful function approximators, we can simply learn a function which maps our independent, normally-distributed $z$ values to whatever latent variables might be needed for the model, and then map those latent variables to $X$.

In the equation, 
$$
P(X|z;\theta) = \mathcal{N}(X|f(z,\theta),\sigma^{2}*I)
$$

If $f(z;\theta)$ is a multilayer neural network then we can imagine the network using its first few layers to map the normally distributed z’s to the latent values (like digit identity, stroke weight, angle, etc.). Then it can use later layers to map those latent values to a fully-rendered digit. In general, we don’t need to worry about ensuring that the latent structure exists. If such latent structure helps the model accurately reproduce (i.e. maximize the likelihood of) the training set, then the network will learn that structure at some layer.


Now all that remains is to maximize Equation,
$$
P(X) = \int P(X|z;\theta)P(z)dz
$$
where $P(z)=\mathcal{N}(z|0,I)$.

As is common in machine learning, if we can find a computable formula for P(X), and we can take the gradient of that formula, then we can optimize the model using GD. 

> It is actually conceptually straightforward to compute P(X) approximately: we first sample a large number of $z$ values ${z1, ..., zn}$, and compute $P(X) ≈ 1/n \sum_{i} P(X|z_{i})$.The problem here is that in high dimensional spaces, $n$ might need to be extremely large before we have an accurate estimate of $P(X)$. 

![](https://i.imgur.com/XySQaRi.png)

To see why, consider our example of handwritten digits. Say that our digit datapoints are stored in pixel space, in 28x28 images as shown in Figure 3. Since $P(X|z)$ is an isotropic Gaussian, the negative log probability of X is proportional squared Euclidean distance between $f(z)$ and $X$. Say that Figure 3(a) is the target $X$ for which we are trying to find $P(X)$. A model which produces the image shown in Figure 3(b) is probably a bad model, since this digit is not much like a 2. Hence, we should set the $σ$ hyperparameter of our Gaussian distribution such that this kind of erroroneous digit does not contribute to $P(X)$. On the other hand, a model which produces Figure 3(c) (identical to $X$ but shifted down and to the right by half a pixel) might be a good model. We would hope that this sample would contribute to $P(X)$. Unfortunately, however, we can’t have it both ways: the squared distance between X and Figure 3(c) is .2693 (assuming pixels range between 0 and 1), but between X and Figure 3(b) it is just .0387. The lesson here is that in order to reject samples like Figure 3(b), we need to set $σ$ very small, such that the model needs to generate something significantly more like X than Figure 3(c)! Even if our model is an accurate generator of digits, we would likely need to sample many thousands of digits before we produce a 2 that is sufficiently similar to the one in Figure 3(a). We might solve this problem by using a better similarity metric, but in practice these are difficult to engineer in complex domains like vision, and they’re difficult to train without labels that indicate which datapoints are similar to each other. Instead, VAEs alter the sampling procedure to make it faster, without changing the similarity metric.


## Setting up Objective 
Is there a shortcut we can take when using sampling to compute Equation 1? In practice, for most $z$, $P(X|z)$ will be nearly zero, and hence contribute almost nothing to our estimate of $P(X)$. The key idea behind the variational autoencoder is to attempt to sample values of $z$ that are likely to have produced X, and compute $P(X)$ just from those. This means that we need a new function $Q(z|X)$ which can take a value of X and give us a distribution over $z$ values that are likely to produce $X$. Hopefully the space of z values that are likely under $Q$ will be much smaller than the space of all $z’s$ that are likely under the prior $P(z)$.  This lets us, for example, compute $E_{z∼Q}P(X|z)$ relatively easily. 
However, if $z$ is sampled from an arbitrary distribution with PDF $Q(z)$, which is not $\mathcal{N}(0, I)$, then how does that help us optimize $P(X)$? The first thing we need to do is relate $E_{z∼Q}P(X|z)$ and $P(X)$. We’ll see where $Q$ comes from later.

The relationship between $E_{z∼Q}P(X|z)$ and $P(X)$ is one of the corner- stones of variational Bayesian methods. We begin with the definition of Kullback-Leibler divergence (KL divergence or $D$) between $P(z|X)$ and $Q(z)$, for some arbitrary $Q$(which may or may not depend on $X$):
$$
D[Q(z) || P(z|X)] = E_{z\sim Q}[logQ(z) - logP(z|X)]
$$

We can get both $P(X)$ and $P(X|z)$ into this equation by applying Bayes rule to $P(z|X)$:

$$
D[Q(z) || P(z|X)] = E_{z\sim Q}[logQ(z) - logP(X|z) - logP(z)] + logP(X)
$$

Here, $logP(X)$ comes out of the expectation because it does not depend on $z$. Negating both sides, rearranging, and contracting part of $E_{z∼Q}$ into a KL-divergence terms yields:
$$
logP(X) - D[Q(z)||P(z|X)] = E_{z \sim Q}[logP(X|z)] - D[Q(z)||P(z)] 
$$

Note that $X$ is fixed, and $Q$ can be any distribution, not just a distribution which does a good job mapping $X$ to the $z’s$ that can produce $X$. Since we’re interested in inferring $P(X)$, it makes sense to construct a $Q$ which does depend on $X$, and in particular, one which makes $D[Q(z) || P(z|X)]$ small: 
$$
logP(X) - D[Q(z|X)||P(z|X)] = E_{z \sim Q}[logP(X|z)] - D[Q(z|X)||P(z)] 
$$

This equation serves is the core of the variational autoencoder: 
-  The left hand side has the quantity we want to maximize: $logP(X)$ (plus an error term, which makes Q produce $z’s$ that can reproduce a given $X$; this term will become small if $Q$ is high-capacity).
-  The right hand side is something we can optimize via stochastic gradient descent given the right choice of $Q$ (although it may not be obvious yet how).

>The framework—in particular, the right hand side of equation has suddenly taken a form which looks like an autoencoder, since $Q$ is “encoding” $X$ into $z$, and $P$ is “decoding” it to reconstruct $X$. 

:::info 
Starting with the left hand side, we are maximizing log $P(X)$ while simultaneously minimizing $D[Q(z|X)||P(z|X)]$. $P(z|X)$ is not something we can compute analytically: it describes the values of z that are likely to give rise to a sample like X under our model. However, the second term on the left is pulling $Q(z|x)$ to match $P(z|X)$. Assuming we use an arbitrarily high-capacity model for $Q(z|x)$, then $Q(z|x)$ will hopefully actually match $P(z|X)$, in which case this KL- divergence term will be zero, and we will be directly optimizing $logP(X)$.
:::

## Optimizing the Objective

- First we need to be a bit more specific about the form that $Q(z|X)$ will take. The usual choice is to say that $Q(z|X)=N(z|μ(X;θ),Σ(X;θ))$, where $μ$ and $Σ$ are arbitrary deterministic functions with parameters $θ$ that can be learned from data (we will omit $θ$ in later equations). In practice, $μ$ and $Σ$ are again implemented via neural networks, and $Σ$ is constrained to be a diagonal matrix. The advantages of this choice are computational, as they make it clear how to compute the right hand side. The last term—$D[Q(z|X)||P(z)]$—is now a KL-divergence between two multivariate Gaussian distributions

- The first term on the right hand side of the objective is a bit more tricky. We could use sampling to estimate $E_{z∼Q}[log P(X|z)]$, but getting a good estimate would require passing many samples of $z$ through $f$, which would be expensive. Hence, as is standard in stochastic gradient descent, we take one sample of $z$ and treat $P(X|z)$ for that $z$ as an approximation of $E_{z∼Q}[log P(X|z)]$. After all, we are already doing stochastic gradient descent over different values of X sampled from a dataset D. The full gradient equation we get now is: 
$$
E_{X \sim D}[logP(X) - D[Q(z|X)||P(z|X)]] = E_{X \sim D}[E_{z \sim Q}[logP(X|z)] - D[Q(z|X)||P(z)]]
$$

- If we take the gradient of this equation, the gradient symbol can be moved into the expectations. Therefore, we can sample a single value of $X$ and $a$ single value of $z$ from the distribution $Q(z|X)$, and compute the gradient of:
 $$
     log P(X|z) − D [Q(z|X)||P(z)] .
 $$
We can then average the gradient of this function over  arbitrarily many samples of $X$ and $z$, and the result converges to the gradient of previous equation. 

-  $E_{z \sim Q}[log P(X|z)]$ depends not just on the parameters of $P$, but also on the parameters of $Q$. However, in previous qquation, this dependency has disappeared! In order to make VAEs work, it’s essential to drive Q to produce codes for X that P can reliably decode. To see the problem a different way, the network described in Equation 9 is much like the network shown in Figure 4 (left).
![](https://i.imgur.com/vtrcpQS.png)

- The forward pass of this network works fine and, if the output is averaged over many samples of $X$ and $z$, produces the correct expected value. However, we need to back-propagate the error through a layer that samples $z$ from $Q(z|X)$, which is a non-continuous operation and has no gradient. Stochastic gradient descent via backpropagation can handle stochastic inputs, but not stochastic units within the network! The solution, called the “reparameterization trick". 

:::info
Reparameterization trick: 
Move the sampling to an input layer. Given $μ(X)$ and $Σ(X)$—the mean and covariance of $Q(z|X)$—we can sample from $N(μ(X), Σ(X))$ by first sampling $ε∼N(0, I)$, then computing $z=μ(X)+Σ^{1/2}(X)∗ε$. 
:::
























