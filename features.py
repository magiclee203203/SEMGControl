import numpy as np
import numpy.fft as fft
from scipy import signal
from scipy.integrate import trapezoid
from statsmodels.tsa.ar_model import AutoReg
from config import ARDUINO_SAMPLE_RATE
import pywt


class FeatureExtractor:
    def __init__(self, data: np.ndarray):
        self.data = data

    def get_features(self):
        rms = self.__rms()
        mav = self.__mav()
        wl = self.__wl()
        zc = self.__zc()
        ssc = self.__ssc()
        var = self.__var()
        wa = self.__wa()
        mcv = self.__mcv()
        mpf = self.__mpf()
        ar0, ar1, ar2, ar3, ar4, ar5, ar6 = self.__ar()
        sm2 = self.__sm2()
        mf = self.__mf()
        me = self.__me()
        e0, e1, e2, e3, e4, e5, e6, e7, a0, a1, a2, a3, a4, a5, a6, a7, v0, v1, v2, v3, v4, v5, v6, v7 = self.__wpt()
        return np.hstack((rms, mav, wl, zc, ssc, var, wa, mcv, mpf,
                          ar0, ar1, ar2, ar3, ar4, ar5, ar6, sm2, mf, me,
                          e0, e1, e2, e3, e4, e5, e6, e7,
                          a0, a1, a2, a3, a4, a5, a6, a7,
                          v0, v1, v2, v3, v4, v5, v6, v7))

    def __rms(self):
        return np.sqrt(np.mean(self.data ** 2, axis=0))

    def __mav(self):
        return np.mean(np.abs(self.data), axis=0)

    def __wl(self):
        return np.sum(np.abs(np.diff(self.data, axis=0)), axis=0) / self.data.shape[0]

    def __zc(self, threshold=10e-7):
        num_of_zc = []
        channel = self.data.shape[1]
        length = self.data.shape[0]

        for i in range(channel):
            count = 0
            for j in range(1, length):
                diff = self.data[j, i] - self.data[j - 1, i]
                mult = self.data[j, i] * self.data[j - 1, i]

                if np.abs(diff) > threshold and mult < 0:
                    count = count + 1
            num_of_zc.append(count / length)
        return np.array(num_of_zc)

    def __ssc(self, threshold=10e-7):
        num_of_ssc = []
        channel = self.data.shape[1]
        length = self.data.shape[0]

        for i in range(channel):
            count = 0
            for j in range(2, length):
                diff1 = self.data[j, i] - self.data[j - 1, i]
                diff2 = self.data[j - 1, i] - self.data[j - 2, i]
                sign = diff1 * diff2

                if sign < 0:
                    if np.abs(diff1) > threshold or np.abs(diff2) > threshold:
                        count = count + 1
            num_of_ssc.append(count / length)

        return np.array(num_of_ssc)

    def __var(self):
        return np.var(self.data, axis=0)

    def __wa(self, threshold=10e-7):
        return np.sum(np.where(np.diff(self.data, axis=0) > threshold, 1, 0), axis=0) / self.data.shape[0]

    def __mcv(self):
        return np.sum(self.data ** 3, axis=0) / self.data.shape[0]

    def __mpf(self):
        pxx_all = []
        for i in range(self.data.shape[1]):
            f, pxx = signal.welch(self.data[:, i], ARDUINO_SAMPLE_RATE, nperseg=200)
            mean_power = trapezoid(pxx, f)
            mean_frequency = trapezoid(f * pxx, f) / mean_power
            pxx_all.append(mean_frequency)
        pxx_all = np.array(pxx_all)
        return pxx_all

    def __ar(self):
        order = 7
        row = []

        for j in range(self.data.shape[1]):
            x = self.data[:, j]
            model = AutoReg(x, order).fit()
            params = model.params[1:order + 1]
            row.append(params)
        row = np.array(row)
        return row[:, 0], row[:, 1], row[:, 2], row[:, 3], row[:, 4], row[:, 5], row[:, 6]

    def __sm2(self):
        pxx_all = []
        for i in range(self.data.shape[1]):
            f, pxx = signal.welch(self.data[:, i], ARDUINO_SAMPLE_RATE, nperseg=200)
            mean_frequency = trapezoid(f * f * pxx, f)
            pxx_all.append(mean_frequency)
        pxx_all = np.array(pxx_all)
        return pxx_all

    def __mf(self):
        pxx_all = []
        for i in range(self.data.shape[1]):
            f, pxx = signal.welch(self.data[:, i], ARDUINO_SAMPLE_RATE, nperseg=200)
            mean_power = trapezoid(pxx, f)
            pxx_all.append(mean_power / 2)
        pxx_all = np.array(pxx_all)
        return pxx_all

    def __me(self):
        return np.mean(abs(fft.fft(self.data)), axis=0)

    def __wpt(self):
        e = []
        a = []
        v = []
        for i in range(self.data.shape[1]):
            wp = pywt.WaveletPacket(data=self.data[:, i], wavelet='db3', mode='symmetric', maxlevel=3)
            n = 3
            re = []  # 第n层所有节点的分解系数
            for j in [node.path for node in wp.get_level(n, 'freq')]:
                re.append(wp[j].data)
            # 第n层能量特征
            energy = []
            for j in re:
                energy.append(pow(np.linalg.norm(j, ord=None), 2))
            average = []
            # 第n层均值
            for j in re:
                average.append(np.average(j))
            var = []
            for j in re:
                var.append(np.var(j))
            a.append(average)
            e.append(energy)
            v.append(var)
        a = np.array(a)
        e = np.array(e)
        v = np.array(v)
        return (e[:, 0], e[:, 1], e[:, 2], e[:, 3], e[:, 4], e[:, 5], e[:, 6], e[:, 7],
                a[:, 0], a[:, 1], a[:, 2], a[:, 3], a[:, 4], a[:, 5], a[:, 6], a[:, 7],
                v[:, 0], v[:, 1], v[:, 2], v[:, 3], v[:, 4], v[:, 5], v[:, 6], v[:, 7])
