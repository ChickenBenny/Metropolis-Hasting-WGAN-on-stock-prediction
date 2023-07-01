This repository try to use Metropolis-Hastings Algorithm and WGAN to do the stock prediction.
## :dash: Quick start
1. Clone the repositroy and enter the folder
```
$ git clone git@github.com:ChickenBenny/Metropolis-Hasting-WGAN-on-stock-prediction.git
$ cd Metropolis-Hasting-WGAN-on-stock-prediction
```
2. Set up the virtual environment
```
$ python -m venv venv
```
* Windows
```
$ venv\Scripts\activate
```
* Mac / Linux
```
$ source venv/bin/activate
```
3. Install package
```
$ pip install -r requirements.txt
```
4. Run the demo notebook. However, GAN is an unstable model, so you may need to run it for a longer period of time or adjust the hyperparameters to obtain better results.


## 🔽 Reference
1. Metropolis-Hastings GAN : https://arxiv.org/abs/1811.11357
2. Wasserstein GAN : https://arxiv.org/abs/1701.07875
3. Repo from borisbanushev :
https://github.com/borisbanushev/stockpredictionai

## 🌟 Idea
Incorporate the ideas from Boris Banushev's repository, as it focuses on implementing GAN-based models for stock prediction. These models leverage the power of GANs' ability to generate realistic and diverse samples, making them suitable for handling stock market dynamics, especially during high-volatility situations.
1. Try using GAN to predict stock prices and simulate the stock distribution.
2. Try enhancing the sampling process using the Metropolis-Hastings algorithm to achieve better convergence and explore a broader range of parameter values.
3. Attempt to improve the prediction accuracy by employing a VAE to extract latent variables and enhance the prediction effect.

## 🖥️ Result 
![](https://hackmd.io/_uploads/H1KioE6O3.png)
* RMSE and MAE from testing dataset


    | RMSE | MAE |
    | -------- | -------- | 
    | 2.077     | 1.673     | 

## License

MIT © ChickenBenny