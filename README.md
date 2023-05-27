# The Value of Out-of-distribution Data

Generalization error always improves with more in-distribution data. However, what happens as we add out-of-distribution (OOD) is still an open question. 
Intuitively, if OOD data is quite different, it seems more data would harm generalization error, though if the OOD data are sufficiently similar, 
much empirical evidence suggests that OOD data can actually improve generalization error. But in our ICML 2023 paper, we show a counter-intuitive phenomenon:
**the generalization error of a task can be a non-monotonic function of the amount of OOD data**.

In particular, we prove that generalization error can improve with small amounts of OOD data, and then get worse than no OOD data with larger amounts.
We analytically demonstrate these results via  Fisher's Linear Discriminant on synthetic datasets, and empirically demonstrate them via deep networks on 
computer vision benchmarks such as MNIST, CIFAR-10, CINIC-10, PACS and DomainNet. Moreover, in the idealistic setting where we know which samples are OOD, 
we show that these non-monotonic trends can be exploited using an appropriately weighted objective of the target and OOD empirical risk.

<p align="center">
<img src="https://github.com/Laknath1996/value-of-ood-data/blob/master/assets/1-summary-plot.png" width="600">
</p>

<p align="center">
<img src="https://github.com/Laknath1996/value-of-ood-data/blob/master/assets/9-simdata-plot.png" width="600">
</p>

<p align="center">
<img src="https://github.com/Laknath1996/value-of-ood-data/blob/master/assets/8-realdata-plot.png" width="600">
</p>

If you use this code/paper for your research, please consider citing,

```
@article{de2022value,
  title={The Value of Out-of-Distribution Data},
  author={De Silva, Ashwin and Ramesh, Rahul and Priebe, Carey E and Chaudhari, Pratik and Vogelstein, Joshua T},
  journal={arXiv preprint arXiv:2208.10967},
  year={2022}
}
```


