# pyright: reportMissingImports=false, reportMissingModuleSource=false
from logging import error
import xarray as xr
import numpy as np
import os 
import logging
import xskillscore

def xarr_std(x):
    return (x - x.mean()) / x.std()

def time_valid_errors(x, x_obs):
    x['date'] = x['date']+x['time'].astype("timedelta64[h]")
    x = x.rename({'date':'valid_time'}).transpose("valid_time","latitude", "longitude") 
    return np.abs(x.msl - x_obs)

class stats:
    def __init__(self, ds, path, obs_path, stats_path, save=True, run_all=True, crps_dim=[]):
        
        self.ds = ds
        self.ds_var = [n for n in ds][0]
        self.comp = dict(zlib=True, complevel=5)
        if self.ds_var == 'tcc':
            self.ds/=100
        self.path = path
        self.set_obs_path(obs_path)
        self.stats_path = f'{stats_path}{self.ds_var}'
        if os.path.exists(f"{self.stats_path}/vss_{self.ds_var}_{str(self.ds['time'].values.astype('datetime64[D]'))}"):
            logging.error(f"{self.stats_path}/vss_{self.ds_var}_{str(self.ds['time'].values.astype('datetime64[D]'))} exists, skipping")
            pass
        else:
            self.load_obs(obs_path)
            self.obs_var = [n for n in self.obs][0]
            self.swap_time_dim()
            self.swap_obs_time_dim()
            valid_filter = self.obs_subset()
            if valid_filter:
                pass
            else:
                logging.error('no dates in obs')
            _ = self.fcst_subset()
            if run_all:
                if valid_filter:
                    stat_ds = self.valid_sample_space()
                    try:
                        stat_ds['crps_ens'] = self.crps_ensemble(crps_dim)
                    except ValueError as e:
                        logging.error(e)
                    stat_ds['me'] = self.mean_bias()
                    encoding= {var: self.comp for var in stat_ds.data_vars}
                    if save:
                        try:
                            stat_ds.to_netcdf(f"{self.stats_path}/vss_{self.ds_var}_{str(self.ds['time'].values.astype('datetime64[D]'))}.nc",encoding=encoding)
                        except OSError as e:
                            logging.error(e)
                            pass

    def swap_time_dim(self,original_dim='step',new_dim='valid_time'):
        try:
            self.ds = self.ds.swap_dims({original_dim:new_dim})
        except ValueError as e:
            logging.error(e)
            try:
                self.ds['valid_time'] = self.ds['valid_time'][0]
                self.ds = self.ds.swap_dims({original_dim:new_dim})
            except:
                logging.error('didnt swap dims still')
            
    def crps_ensemble(self, dim):
        in_fcst = self.ds[self.ds_var].chunk({'latitude':len(self.ds['latitude']), 'longitude':len(self.ds['longitude']), 'number':len(self.ds['number']), 'valid_time':1})
        in_obs = self.obs[self.obs_var].chunk({'latitude':len(self.ds['latitude']), 'longitude':len(self.ds['longitude']), 'valid_time':1})
        return xskillscore.crps_ensemble(in_obs, in_fcst, member_dim='number', dim=dim)


    def swap_obs_time_dim(self,original_dim='time',new_dim='valid_time'):
        self.obs = self.obs.rename(({original_dim:new_dim}))
    
    def set_obs_path(self, path):
        self.analysis_dir = os.path.split(path)[0]

    def load_obs(self, obs_path, dask=True):    
        if dask:
            self.obs = xr.open_dataset(f'{obs_path}',chunks='auto')
        else:
            self.obs = xr.open_dataset(f'{obs_path}')
            
    def obs_subset(self):
        if np.any(self.ds.longitude.values > 180):
            if np.any(self.obs.longitude.values > 180):
                pass
            else:
                self.ds['longitude'] = (self.ds['longitude'] + 180) % 360 - 180
        try:
            self.obs = self.obs.where(self.obs['valid_time'].isin([self.ds['valid_time']]),drop=True).where(self.obs['latitude'].isin([self.ds['latitude']]),drop=True).where(self.obs['longitude'].isin([self.ds['longitude']]),drop=True)
            return True
        except OverflowError as e:
            logging.error(e)
            return False
        except ValueError as e:
            logging.error(e)

    def fcst_subset(self):
        if np.any(self.ds.longitude.values > 180):
            if np.any(self.obs.longitude.values > 180):
                pass
            else:
                self.ds['longitude'] = (self.ds['longitude'] + 180) % 360 - 180
        try:
            self.ds = self.ds.where(self.ds['valid_time'].isin([self.obs['valid_time']]),drop=True)\
                .where(self.ds['latitude'].isin([self.obs['latitude']]),drop=True)\
                    .where(self.ds['longitude'].isin([self.obs['longitude']]),drop=True)
            return True
        except OverflowError:
            return False
        except ValueError:
            return False

    def range(self,dim='number'):
        return self.ds.max(dim=dim) - self.ds.min(dim=dim)

    def mean_bias(self):
        return (self.ds[self.ds_var]-self.obs[self.obs_var]).mean('number')
    
    def valid_sample_space(self):

        valid_grid = xr.ufuncs.logical_and(self.obs[self.obs_var]<=self.ds[self.ds_var].max(dim='number'),self.obs[self.obs_var]>=self.ds[self.ds_var].min(dim='number'))
        try:
            encoding= {var: self.comp for var in valid_grid.data_vars}
        except AttributeError:
            valid_grid = valid_grid.to_dataset(name=f'{self.ds_var}_vss')
            encoding= {var: self.comp for var in valid_grid.data_vars}
        return valid_grid