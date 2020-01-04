r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers

def part1_rnn_hyperparams():
    hypers = dict(
        batch_size=256, seq_len=64,
        h_dim=512, n_layers=3, dropout=0.5,
        learn_rate=0.001, lr_sched_factor=0.5, lr_sched_patience=2,
    )
    # TODO: Set the hyperparameters to train the model.
    # ====== YOUR CODE: ======
    #raise NotImplementedError()
    # ========================
    return hypers


def part1_generation_params():
    #start_seq = ""
    #temperature = .0001
    # TODO: Tweak the parameters to generate a literary masterpiece.
    # ====== YOUR CODE: ======
    # raise NotImplementedError()
    # ========================
    start_seq, temperature = "ACT I.", 0.5
    return start_seq, temperature


part1_q1 = r"""
In this model we want the network to be able to see the connection of the words through time.
If we used the whole text, the sequence would be really long, and that would make it harder, and probably
very difficult for out network to observe the connection between the begining of the text to it's end.
And so, using shorter sequences helps to deal with this problem.   
"""

part1_q2 = r"""
Our network's architecture is built so that between 2 batches the hidden states remain,
and so they act as a kind of memory.
Thus the memory is longer than the sequence length.
"""

part1_q3 = r"""
There is a meaning to the order of the words.
The order of the words in a text is what gives the text it's meaning.
And so, shuffling the order of the batches means some shuffling in sentences words, so meaning would change completely.
As we said in Q2, the hidden states are the memory of the system,
and so if we shuffle the order of the batches we would cause the memory to be useless.
"""

part1_q4 = r"""
1. We want to prevent overfitting when we train out model, and temperature=1.0 is helps us do that.
It gives some "randomness" to our model, which helps the training process.
When we sample the model, we want lower variance, so the words that the out model produces would be connected 
to the words who came before them.

2. Using a high temperatures got us a lot of gibberish and spelling errors, and got us a pretty random text.
This could be explained mathematically, when T is high than (for every k):
    exp(y_k/t)=\~exp(y/T)=\~exp(0)==1.
This means that we will get a uniform probability to all the possible characters,
which really would make a random text. 

3. Using a low temperatures got us diffrent results from the last question.
We didn't get any mistakes, but there are a lot of repeating phrases.
It seems like here the variance is low, so we choose words based on how common they are in our text.
"""
# ==============


# ==============
# Part 2 answers

PART2_CUSTOM_DATA_URL = None


def part2_vae_hyperparams():
    hypers = dict(
        batch_size=0,
        h_dim=0, z_dim=0, x_sigma2=0,
        learn_rate=0.0, betas=(0.0, 0.0),
    )
    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======
    #raise NotImplementedError()
    hypers['batch_size']=64
    hypers['h_dim']=256
    hypers['z_dim']=16
    hypers['x_sigma2']=0.2
    hypers['learn_rate']=0.0005
    hypers['betas']=(0.9,0.9)
    # ========================
    return hypers


part2_q1 = r"""
**Your answer:**


The $\sigma^2$ determine the ratio between the data-loss and the KLD-loss. meaning controlls what loss we want to minimize more.
For higher values the loss will be mostly determine by the KLD-loss, so we probably will get 'more random' images. For lower values,
we will probably will start to over-fit. When experimenting with the hyperparams we saw that with smaller $\sigma^2$ we got faster to 'George Bush' images.

"""

part2_q2 = r"""
**Your answer:**

 1.Reconstruction loss:
    The reconstruction loss purpose is to make the model to train from the given images.
    This is wanted since we want our model to generate images similar to the ones in the dataset

    KL divergence loss:
    The KL Div loss purpose is to try to make the latent space distribution to be close to some informative optimal distribution

 2.KL Div affects the latent space distribution by adjusting z_mu and z_sigma_2. 
     When the model is not close enough to the normal N(0, I) distribution.

3. We benefit from this method because the model learns the distribuion of the latent space(z) given some x from 
     the distribution of x: p(x). I.e p(z|X=x).Considering that at first we didn't know anything about this distribuion
     this is very surprising.
 

"""

# ==============

# ==============
# Part 3 answers

PART3_CUSTOM_DATA_URL = None


def part3_gan_hyperparams():
    hypers = dict(
        batch_size=0, z_dim=0,
        data_label=0, label_noise=0.0,
        discriminator_optimizer=dict(
            type='',  # Any name in nn.optim like SGD, Adam
            lr=0.0,
            # You an add extra args for the optimizer here
        ),
        generator_optimizer=dict(
            type='',  # Any name in nn.optim like SGD, Adam
            lr=0.0,
            # You an add extra args for the optimizer here
        ),
    )
    # TODO: Tweak the hyperparameters to train your GAN.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
    return hypers


part3_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============


