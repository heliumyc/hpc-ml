{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "import matplotlib\n",
    "import matplotlib.pyplot as plt\n",
    "import numpy as np"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 97,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[1.738176020325546,\n",
       " 1.8740326967277579,\n",
       " 5.203279781994832,\n",
       " 2.959520144874502,\n",
       " 8.84967515200664,\n",
       " 2.829395673163558,\n",
       " 0.005274802597206,\n",
       " 0.005329106917286,\n",
       " 8.436488764703528,\n",
       " 2.660318783746622]"
      ]
     },
     "execution_count": 97,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "bandwidth = [6.952704, 7.496131, \n",
    "             20.813119, 11.838081,\n",
    "             35.398701, 11.317583,\n",
    "             0.021099, 0.021316,\n",
    "             33.745955, 10.641275]\n",
    "flop = [1738176020.325546, 1874032696.727758,\n",
    "        5203279781.994832 , 2959520144.874502,\n",
    "        8849675152.006639, 2829395673.163558,\n",
    "        5274802.597206, 5329106.917286,\n",
    "        8436488764.703528, 2660318783.746622]\n",
    "gflop = [x/1e9 for x in flop]\n",
    "gflop"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 98,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[0.25000000292340163,\n",
       " 0.2499999928933683,\n",
       " 0.2500000015372435,\n",
       " 0.24999999111971796,\n",
       " 0.2499999972317244,\n",
       " 0.24999999321087885,\n",
       " 0.25000249287672405,\n",
       " 0.2500050158231375,\n",
       " 0.2500000004357123,\n",
       " 0.25000000317129495]"
      ]
     },
     "execution_count": 98,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arith = [x[1] / x[0] for x in zip(bandwidth, gflop)]\n",
    "arith"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 99,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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gqryww8/EBQUlH46LM2tt97K2bNnLzpFdfr06SvGExYWRnx8fHp/yKeffkqbNm3Sp8+cOTP93+bNmwOwfft2mjVrxhtvvMH1119/UaIrSBx1n4lxpr8u+itg9yI4UV7b9vvvv+fTTz8lPDyciAjXVWFvv/02t99+O4MGDeKee+5hwoQJVKtWjc8++wyA22+/nW+++YaQkBACAwOZNGnSn9YrIsyePZvnn3+ed955h+uvv56SJUvy73//O8t4ihcvzqRJk7j77rtJSUmhadOmF3XiHz16lAYNGlCsWDGmT58OwMCBA4mLi0NViY6OpmHDhrnaF75mtbmM30t7El9YuTAfR2I87Wpq2+rVq7N27VrKlSvn61AyZbW5jONdDV80VytrW+ewZGL83rL4ZQC0qd7mCnOaguZqatv4+Hhfh+BVjk0mCxcu5J///CeffvopVapUYd68eQwZMoQZM2ZQoUIF5s6dy7Bhw/jiiy8oV64cs2bNYuTIkXz11VcEBQUxc+ZMxowZwzfffENgYCBTp05l/PjxLFiwgCJFijB58mQmT57M0qVLAfj444+ZOXMmCxcuBODDDz9k7ty5fPvttwCMGDGCRYsWpV9HPnToUFavXs2XX34JuO7W3bhxIzNmzADgzTffJDY2lqlTpwLw2muvkZCQkH5+95VXXuHw4cOMGzcOgJdeeokzZ84wevRogPRr2IcPd5WqePrppylRogRDhw4FoH///pQtW5Z//etfADz88MNUqVKFN954A4A+ffoQFhbG3/72NwDuu+8+IiIiGDRoEAA9evSgefPmvPTSS4CrUzM6Opr/+7//A6BTp0507tyZp556CoB27dpx77338thjjwEQFRVF37596du3L8nJybRv355HH32UPn36cPr0aW6//XaefPJJ7r33XgYvHExMTAwTW0+ke/fuJCYm0rNnT1588UU6d+7M/v37ue+++xg0aBAdO3YkISGBBx54gFdffZV27dqxY8cO+vXrxz/+8Q/atGlDbGwsjz/+OG+//TYtWrQgJiaGAQMG8O6779K0aVM2btzIc889x/Dhw4mIiGDNmjUMHDiQUaNGUb9+fVatWsVf//pXPvroI8LCwli2bBmvv/46EydOpGbNmva3l82/vQF/eY2eY3uRfPY0ERutwkFB56iruew+E2ca3WE0Yb/Z6RCnOJucyoq4Q7R9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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "xx = np.arange(0, 200/30, 0.5)\n",
    "plt.plot(xx, xx*30)\n",
    "plt.text(0.5,10, 'max bandwidth \\n 30Gb/s')\n",
    "\n",
    "xx = np.arange(0, 200/30, 0.5)\n",
    "plt.plot(xx, np.ones(len(xx))*200, 'k:')\n",
    "plt.text(10,250,'200 Gflops')\n",
    "\n",
    "xx = np.arange(200/30, 100, 0.5)\n",
    "plt.plot(xx, np.ones(len(xx))*200, 'k')\n",
    "\n",
    "plt.axvline(x=0.25, color='g', linestyle=':')\n",
    "plt.text(0.25,0.5,'0.25')\n",
    "\n",
    "plt.axvline(x=200/30, color='g', linestyle=':')\n",
    "plt.text(200/30,0.5, 'limit 6.67')\n",
    "\n",
    "plt.scatter(arith, gflop, color='r')\n",
    "\n",
    "\n",
    "plt.xscale('log', basex=10)\n",
    "plt.yscale('log', basey=10)\n",
    "plt.ylim(0.01, 1000)\n",
    "plt.xlim(0.01,100)\n",
    "plt.xlabel('Arithmetic intensity FLOPS/BYTES')\n",
    "plt.ylabel('FLOPS GFlop/sec')\n",
    "plt.title('Roofline model')\n",
    "plt.savefig('roofline.png', dpi=800)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
