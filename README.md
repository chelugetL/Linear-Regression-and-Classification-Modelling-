# Linear-Regression-and-Classification-Modelling-
When performing a machine learning task on a small dataset, one often suffers from
the over-fitting problem, where the model accurately remembers all training data,
including noise and unrelated features. Such a model often performs badly on new
test or real data that have not been seen before. Because the model treats the
training data too seriously, it failed to learn any meaningful pattern out of it, but
simply memorizing everything it has seen.
Now, one solution to solve this issue is called regularization. The idea is applying an L1
norm to the solution vector of your machine learning problem (In case of deep
learning, it’s the neural network weights.), and trying to make it as small as possible.
So if your initial goal is finding the best vector x to minimize a loss function f(x), your
new task should incorporate the L1 norm of x into the formula, finding the minimum
(f(x) + L1norm(x)). The big claim they often throw at you is this: An x with small L1
norm tends to be a sparse solution. Being sparse means that the majority of x’s
components (weights) are zeros, only few are non-zeros. And a sparse solution could
avoid over-fitting.
