# encoding: utf-8

import os
import shutil
import argparse
import setproctitle
import scipy.stats
import numpy as np
from collections import Counter
from math import radians, cos, sin, asin, sqrt
from utils import get_gps, read_data_from_file, read_logs_from_file


def geodistance(lng1,lat1,lng2,lat2):
    lng1, lat1, lng2, lat2 = map(radians, [float(lng1), float(lat1), float(lng2), float(lat2)]) 
    dlon=lng2-lng1
    dlat=lat2-lat1
    a=sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    distance=2*asin(sqrt(a))*6371*1000 
    distance=round(distance/1000,3)
    return distance


class EvalUtils(object):
    """
    some commonly-used evaluation tools and functions
    """

    @staticmethod
    def filter_zero(arr):
        """
        remove zero values from an array
        :param arr: np.array, input array
        :return: np.array, output array
        """
        arr = np.array(arr)
        filtered_arr = np.array(list(filter(lambda x: x != 0., arr)))
        return filtered_arr

    @staticmethod
    def arr_to_distribution(arr, min, max, bins):
        """
        convert an array to a probability distribution
        :param arr: np.array, input array
        :param min: float, minimum of converted value
        :param max: float, maximum of converted value
        :param bins: int, number of bins between min and max
        :return: np.array, output distribution array
        """
        distribution, base = np.histogram(
            arr, np.arange(
                min, max, float(
                    max - min) / bins))
        return distribution, base[:-1]

    @staticmethod
    def norm_arr_to_distribution(arr, bins=100):
        """
        normalize an array and convert it to distribution
        :param arr: np.array, input array
        :param bins: int, number of bins in [0, 1]
        :return: np.array, np.array
        """
        arr = (arr - arr.min()) / (arr.max() - arr.min())
        arr = EvalUtils.filter_zero(arr)
        distribution, base = np.histogram(arr, np.arange(0, 1, 1. / bins))
        return distribution, base[:-1]

    @staticmethod
    def log_arr_to_distribution(arr, min=-30., bins=100):
        """
        calculate the logarithmic value of an array and convert it to a distribution
        :param arr: np.array, input array
        :param bins: int, number of bins between min and max
        :return: np.array,
        """
        arr = (arr - arr.min()) / (arr.max() - arr.min())
        arr = EvalUtils.filter_zero(arr)
        arr = np.log(arr)
        distribution, base = np.histogram(arr, np.arange(min, 0., 1./bins))
        ret_dist, ret_base = [], []
        for i in range(bins):
            if int(distribution[i]) == 0:
                continue
            else:
                ret_dist.append(distribution[i])
                ret_base.append(base[i])
        return np.array(ret_dist), np.array(ret_base)

    @staticmethod
    def get_js_divergence(p1, p2):
        """
        calculate the Jensen-Shanon Divergence of two probability distributions
        :param p1:
        :param p2:
        :return:
        """
        # normalize
        p1 = p1 / (p1.sum()+1e-14)
        p2 = p2 / (p2.sum()+1e-14)
        m = (p1 + p2) / 2
        js = 0.5 * scipy.stats.entropy(p1, m) + \
            0.5 * scipy.stats.entropy(p2, m)
        return js
    


class IndividualEval(object):

    def __init__(self, data):
        if data == 'mobile':       
            self.X, self.Y = get_gps('../data/mobile/gps')
            self.max_locs = 8606
            self.max_distance = 2.088
        else:
            self.X, self.Y = get_gps('../data/geolife/gps')
            self.max_locs = 23768
            self.max_distance = 247.3
   

    def get_topk_visits(self,trajs, k):
        topk_visits_loc = []
        topk_visits_freq = []
        for traj in trajs:
            topk = Counter(traj).most_common(k)
            for i in range(len(topk), k):
                # supplement with (loc=-1, freq=0)
                topk += [(-1, 0)]
            loc = [l for l, _ in topk]
            freq = [f for _, f in topk]
            loc = np.array(loc, dtype=int)
            freq = np.array(freq, dtype=float) / trajs.shape[1]
            topk_visits_loc.append(loc)
            topk_visits_freq.append(freq)
        topk_visits_loc = np.array(topk_visits_loc, dtype=int)
        topk_visits_freq = np.array(topk_visits_freq, dtype=float)
        return topk_visits_loc, topk_visits_freq

    
    def get_overall_topk_visits_freq(self, trajs, k):
        _, topk_visits_freq = self.get_topk_visits(trajs, k)
        mn = np.mean(topk_visits_freq, axis=0)
        return mn / np.sum(mn)


    def get_overall_topk_visits_loc_freq_arr(self, trajs, k=1):
        topk_visits_loc, _ = self.get_topk_visits(trajs, k)
        k_top = np.zeros(self.max_locs, dtype=float)
        for i in range(k):
            cur_k_visits = topk_visits_loc[:, i]
            for ckv in cur_k_visits:
                index = int(ckv)
                if index == -1:
                    continue
                k_top[index] += 1
        k_top = k_top / np.sum(k_top)
        return k_top

    
    def get_overall_topk_visits_loc_freq_dict(self, trajs, k):
        topk_visits_loc, _ = self.get_topk_visits(trajs, k)
        k_top = {}
        for i in range(k):
            cur_k_visits = topk_visits_loc[:, i]
            for ckv in cur_k_visits:
                index = int(ckv)
                if index in k_top:
                    k_top[int(ckv)] += 1
                else:
                    k_top[int(ckv)] = 1
        return k_top

    def get_overall_topk_visits_loc_freq_sorted(self, trajs, k):
        k_top = self.get_overall_topk_visits_loc_freq_dict(trajs, k)
        k_top_list = list(k_top.items())
        k_top_list.sort(reverse=True, key=lambda k: k[1])
        return np.array(k_top_list)


    def get_geodistances(self, trajs):
        distances = []
        seq_len = 48
        for traj in trajs:
            for i in range(seq_len - 1):
                lng1 = self.X[traj[i]]
                lat1 = self.Y[traj[i]]
                lng2 = self.X[traj[i + 1]]
                lat2 = self.Y[traj[i + 1]]
                distances.append(geodistance(lng1,lat1,lng2,lat2))
        distances = np.array(distances, dtype=float)
        return distances
    
    def get_distances(self, trajs):
        distances = []
        seq_len = 48
        for traj in trajs:
            for i in range(seq_len - 1):
                dx = self.X[traj[i]] - self.X[traj[i + 1]]
                dy = self.Y[traj[i]] - self.Y[traj[i + 1]]
                distances.append(dx**2 + dy**2 )
        distances = np.array(distances, dtype=float)
        return distances
    
    def get_durations(self, trajs):
        d = []
        for traj in trajs:
            num = 1
            for i, lc in enumerate(traj[1:]):
                if lc == traj[i]:
                    num += 1
                else:
                    d.append(num)
                    num = 1
        return np.array(d)/48
    
    def get_gradius(self, trajs):
        """
        get the std of the distances of all points away from center as `gyration radius`
        :param trajs:
        :return:
        """
        gradius = []
        seq_len = 48
        for traj in trajs:
            xs = np.array([self.X[t] for t in traj])
            ys = np.array([self.Y[t] for t in traj])
            xcenter, ycenter = np.mean(xs), np.mean(ys)
            dxs = xs - xcenter
            dys = ys - ycenter
            rad = [dxs[i]**2 + dys[i]**2 for i in range(seq_len)]
            rad = np.mean(np.array(rad, dtype=float))
            gradius.append(rad)
        gradius = np.array(gradius, dtype=float)
        return gradius
    
    def get_periodicity(self, trajs):
        """
        stat how many repetitions within a single trajectory
        :param trajs:
        :return:
        """
        reps = []
        for traj in trajs:
            reps.append(float(len(set(traj)))/48)
        reps = np.array(reps, dtype=float)
        return reps

    def get_timewise_periodicity(self, trajs):
        """
        stat how many repetitions of different times
        :param trajs:
        :return:
        """
        pass


    def get_geogradius(self, trajs):
        """
        get the std of the distances of all points away from center as `gyration radius`
        :param trajs:
        :return:
        """
        gradius = []
        for traj in trajs:
            xs = np.array([self.X[t] for t in traj])
            ys = np.array([self.Y[t] for t in traj])
            lng1, lat1 = np.mean(xs), np.mean(ys)
            rad = []
            for i in range(len(xs)):                   
                lng2 = xs[i]
                lat2 = ys[i]
                distance = geodistance(lng1,lat1,lng2,lat2)
                rad.append(distance)
            rad = np.mean(np.array(rad, dtype=float))
            gradius.append(rad)
        gradius = np.array(gradius, dtype=float)
        return gradius

    def get_individual_jsds(self, t1, t2):
        """
        get jsd scores of individual evaluation metrics
        :param t1: test_data
        :param t2: gene_data
        :return:
        """
        d1 = self.get_distances(t1)
        d2 = self.get_distances(t2)
        
        d1_dist, _ = EvalUtils.arr_to_distribution(
            d1, 0, self.max_distance, 10000)
        d2_dist, _ = EvalUtils.arr_to_distribution(
            d2, 0, self.max_distance, 10000)
        d_jsd = EvalUtils.get_js_divergence(d1_dist, d2_dist)
        

        g1 = self.get_gradius(t1)
        g2 = self.get_gradius(t2)
        g1_dist, _ = EvalUtils.arr_to_distribution(
            g1, 0, self.max_distance**2, 10000)
        g2_dist, _ = EvalUtils.arr_to_distribution(
            g2, 0, self.max_distance**2, 10000)
        g_jsd = EvalUtils.get_js_divergence(g1_dist, g2_dist)
        
        
        
        du1 = self.get_durations(t1)
        du2 = self.get_durations(t2)     
        du1_dist, _ = EvalUtils.arr_to_distribution(du1, 0, 1, 48)
        du2_dist, _ = EvalUtils.arr_to_distribution(du2, 0, 1, 48)
        du_jsd = EvalUtils.get_js_divergence(du1_dist, du2_dist)
        
        p1 = self.get_periodicity(t1)
        p2 = self.get_periodicity(t2)
        p1_dist, _ = EvalUtils.arr_to_distribution(p1, 0, 1, 48)
        p2_dist, _ = EvalUtils.arr_to_distribution(p2, 0, 1, 48)
        p_jsd = EvalUtils.get_js_divergence(p1_dist, p2_dist)

        
        l1 =  CollectiveEval.get_visits(t1,self.max_locs)
        l2 =  CollectiveEval.get_visits(t2,self.max_locs)
        l1_dist, _ = CollectiveEval.get_topk_visits(l1, 100)
        l2_dist, _ = CollectiveEval.get_topk_visits(l2, 100)
        l1_dist, _ = EvalUtils.arr_to_distribution(l1_dist,0,1,100)
        l2_dist, _ = EvalUtils.arr_to_distribution(l2_dist,0,1,100)
        l_jsd = EvalUtils.get_js_divergence(l1_dist, l2_dist)

        f1 = self.get_overall_topk_visits_freq(t1, 100)
        f2 = self.get_overall_topk_visits_freq(t2, 100)
        f1_dist, _ = EvalUtils.arr_to_distribution(f1,0,1,100)
        f2_dist, _ = EvalUtils.arr_to_distribution(f2,0,1,100)
        f_jsd = EvalUtils.get_js_divergence(f1_dist, f2_dist)


        return d_jsd,  g_jsd,  du_jsd,  p_jsd, l_jsd, f_jsd




class CollectiveEval(object):
    """
    collective evaluation metrics
    """
    @staticmethod
    def get_visits(trajs,max_locs):
        """
        get probability distribution of visiting all locations
        :param trajs:
        :return:
        """
        visits = np.zeros(shape=(max_locs), dtype=float)
        for traj in trajs:
            for t in traj:
                visits[t] += 1
        visits = visits / np.sum(visits)
        return visits

    @staticmethod
    def get_timewise_visits(trajs):
        """
        stat how many visits of a certain location in a certain time
        :param trajs:
        :return:
        """
        pass

    @staticmethod
    def get_topk_visits(visits, K):
        """
        get top-k visits and the corresponding locations
        :param trajs:
        :param K:
        :return:
        """
        locs_visits = [[i, visits[i]] for i in range(visits.shape[0])]
        locs_visits.sort(reverse=True, key=lambda d: d[1])
        topk_locs = [locs_visits[i][0] for i in range(K)]
        topk_probs = [locs_visits[i][1] for i in range(K)]
        return np.array(topk_probs), topk_locs

    @staticmethod
    def get_topk_accuracy(v1, v2, K):
        """
        get the accuracy of top-k visiting locations
        :param v1:
        :param v2:
        :param K:
        :return:
        """
        _, tl1 = CollectiveEval.get_topk_visits(v1, K)
        _, tl2 = CollectiveEval.get_topk_visits(v2, K)
        coml = set(tl1) & set(tl2)
        return len(coml) / K


def evaluate(datasets):
    if datasets == 'telecom':
        individualEval = IndividualEval(data='mobile')
        start_point = np.load('../data/mobile/start.npy')
    else:
        individualEval = IndividualEval(data='geolife')
        start_point = np.load('../data/geolife/start.npy')    
    
    test_data = read_data_from_file('../data/%s/test.data' % opt.datasets)
    gene_data = read_data_from_file('../data/%s/gene.data' % opt.datasets)
    print(individualEval.get_individual_jsds(test_data,gene_data))
    



if __name__ == "__main__":
    # global
    parser = argparse.ArgumentParser()
    parser.add_argument('--task',default='default', type=str)
    parser.add_argument('--cuda',default=0,type=int)
    parser.add_argument('--datasets',default='geolife',type=str)
    opt = parser.parse_args()
    if opt.datasets == 'mobile':
        max_locs = 8606
    else:
        max_locs = 23768
    evaluate(opt.datasets)
