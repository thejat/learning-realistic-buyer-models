### Code for the paper "Learning Buyer Behavior through Pricing"

This codebase is for algorithms proposed in the aforementioned [paper](http://arxiv.org/#). It consists of four python scripts:

  * algorithms.py : has the proposed and competitor algorithms
  * data.py : generates data (e.g., from the billion prices project)
  * experiments.py : (calls algorithms on buyer models)
  * buyers.py : has the four buyer models

The easiest way to get started is to look at experiments.py and go from there.

###### Some useful pointers:

To reload Ipython modules (useful for debugging) following [this](https://stackoverflow.com/questions/5364050/reloading-submodules-in-ipython):

```python
%load_ext autoreload
%autoreload 2
```

This can be saved in `~/.ipython/profile_default/ipython_config.py` as

```python
c.InteractiveShellApp.extensions = ['autoreload']     
c.InteractiveShellApp.exec_lines = ['%autoreload 2']
```