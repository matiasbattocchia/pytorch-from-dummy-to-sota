# ## 3. A great foundation to build on top of

# PyTorch it self is great! But it is not alone... Is is the foundation of a rich ecosystem of tools and libraries that makes PyTorch even greater.

# Introducing...
#
# ![fastai logo](img/fastai.png)

# The [fastai library](https://docs.fast.ai/) simplifies training fast and accurate neural nets using modern best practices. Fastai sits on top of PyTorch and it's based on research in to deep learning best practices undertaken at [fast.ai](https://www.fast.ai), including "out of the box" support for vision, text, tabular, and collab (collaborative filtering) models.

# In order to install fastai, you can follow these [installation instruction](https://github.com/fastai/fastai/blob/master/README.md#installation).

# In[67]:


from fastai import *
from fastai.vision import *


# In[68]:


PETS_PATH = Path('datasets/oxford-iiit-pet/images').expanduser()


# ### 3.1. Data

# We will use the Oxford-IIIT Pet dataset, that contains images of 37 breeds of dogs and cats.

# Fastai's `data_block` allows to load data, split train/valid sets, label, transform and normalize in just one step.

# In[69]:


data = (ImageList.from_folder(PETS_PATH)
        .split_by_rand_pct(0.1, seed=7)
        .label_from_func(lambda fn: fn.name.rsplit('_', 1)[0])
        .transform(get_transforms(), size=354)
        .databunch(bs=32)
        .normalize(imagenet_stats))
data.show_batch(rows=3, figsize=(6,6))


# ### 3.2. Model & train

# We can create a state-of-the-art model in just one line.

# In[70]:


learn = cnn_learner(data, models.resnet50, metrics=accuracy)


# It also helps us picking the optimal learning rate (you can read more about this technique on fastai's documentation).

# In[71]:


#learn.lr_find()
#learn.recorder.plot()


# Again, train the model requires just one line of code.

# In[ ]:


learn.fit_one_cycle(3, 3e-3)


# In[ ]:


learn.save('stage-1')


# Optionally, we can fine-tune our model easily.

# In[8]:


learn.unfreeze()
learn.fit_one_cycle(2, slice(1e-6, 1e-4))


# WOW! It doesn't sound like a trivial task to classify among 37 breads of dogs and cats with a 95% accuracy... I guess this model can outperform most humans on this task!

# In[9]:


learn.save('stage-2')


# In[11]:


#learn.load('stage-2');


# If we visualize the model, we can see that it is pure PyTorch.

# In[12]:


#learn.model


# ### 3.3. Predict

# Let's try with Bruno, my dog.

# In[13]:


#img = open_image('img/bruno.jpg',)
#img.show(figsize=(6,6))


# In[14]:


#cat,y,probs = learn.predict(img)
#cat, probs[y]


# Great! Bruno is indeed a beagle.

# ## Takeaways
#
# - PyTorch provides a pythonic way of doing Deep Learning.
# - Easy, flexible & powerful.
# - Fullstack Deep Learning (no magic).
# - Fastai is awesome!

# ## End
