# fastpredict

Has the following benefits over using fastai directly:
- cleaner API for batch prediction
- takes care of using CPU or GPU if you specify in the constructor

How to use:

```python
from fastpredict import FastPredictBool
f = FastPredictBool.from_path('/path/to/learner.pkl')
score = f.predict_path('/path/to/image.jpg')
scores = f.batch_predict_paths(['/path/to/image1.jpg', '/path/to/image2.jpg'])
```

To prune a model:
```python
from fastpredict import FastPredict
from fastai.vision.all import load_learner
learn = load_learner('/path/to/current/learner.pkl')
new_learn = FastPredict.clean_learner(learn)
new_learn.export('/path/to/new/learner.pkl')
```

