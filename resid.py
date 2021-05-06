def plot_diagnostics(self, variable=0, lags=40, fig=None,
                     figsize=(15,7), savefig = False, title = None, path = None):
  from statsmodels.graphics.utils import _import_mpl, create_mpl_fig
  _import_mpl()
  fig = create_mpl_fig(fig, figsize)

  # # Eliminate residuals associated with burned or diffuse likelihoods
  # d = np.maximum(self.loglikelihood_burn, self.nobs_diffuse)
  # resid = self.filter_results.standardized_forecasts_error[variable, d:]
  # loglikelihood_burn: the number of observations during which the likelihood is not evaluated.

  # Standardize residual
  # Source: https://alkaline-ml.com/pmdarima/1.1.1/_modules/pmdarima/arima/arima.html
  resid = self
  resid = (resid - np.nanmean(resid)) / np.nanstd(resid)

  # Top-left: residuals vs time
  ax = fig.add_subplot(221)
  if hasattr(self.data, 'dates') and self.data.dates is not None:
      x = self.data.dates[d:]._mpl_repr()
  else:
      x = np.arange(len(resid))
  ax.plot(x, resid)
  ax.hlines(0, x[0], x[-1], alpha=0.5)
  ax.set_xlim(x[0], x[-1])
  ax.set_title('Standardized residual')

  # Top-right: histogram, Gaussian kernel density, Normal density
  # Can only do histogram and Gaussian kernel density on the non-null
  # elements
  resid_nonmissing = resid[~(np.isnan(resid))]
  ax = fig.add_subplot(222)

  # gh5792: Remove  except after support for matplotlib>2.1 required
  try:
      ax.hist(resid_nonmissing, density=True, label='Hist')
  except AttributeError:
      ax.hist(resid_nonmissing, normed=True, label='Hist')

  from scipy.stats import gaussian_kde, norm
  kde = gaussian_kde(resid_nonmissing)
  xlim = (-1.96*2, 1.96*2)
  x = np.linspace(xlim[0], xlim[1])
  ax.plot(x, kde(x), label='KDE')
  ax.plot(x, norm.pdf(x), label='N(0,1)')
  ax.set_xlim(xlim)
  ax.legend()
  ax.set_title('Histogram plus estimated density')

  # Bottom-left: QQ plot
  ax = fig.add_subplot(223)
  from statsmodels.graphics.gofplots import qqplot
  qqplot(resid_nonmissing, line='s', ax=ax)
  ax.set_title('Normal Q-Q')

  # Bottom-right: Correlogram
  ax = fig.add_subplot(224)
  from statsmodels.graphics.tsaplots import plot_pacf
  plot_pacf(resid, ax=ax, lags=lags)
  ax.set_title('Partial Autocorrelation function')
 
  ax.set_ylim(-0.1, 0.1)

  if savefig == True:
    fig.suptitle('Residual diagnostic for Sinusoidal model', fontsize = 20)
    fig.savefig(path + title + '.jpg', dpi = 500)
    fig.show()
  return fig
