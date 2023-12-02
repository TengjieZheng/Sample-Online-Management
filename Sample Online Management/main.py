import numpy as np
from matplotlib import pyplot as plt
from SampleManger import SampleManager
from SampleGeneration import Sample_generation
import pickle

"""
Test of the sample online management algorithm
"""

class Test():
    def __init__(self):
        self.file_name = 'data.pkl'
        self.dim_input = 2
        self.sample_generation = Sample_generation(input_dimension=self.dim_input, sigma_noise=0.01, flag_record=True, type=2)
        self.sm = SampleManager(capacity=500, input_dimension=self.dim_input, output_dimension=1, rho=0.5)
        self.rhoList = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        self.smList = []
        for rho in self.rhoList:
            self.smList.append( SampleManager(capacity=500, input_dimension=self.dim_input, output_dimension=1, rho=rho) )

    def run(self):
        for ii in range(200000):
            input, output = self.sample_generation.new_sample()
            self.sm.update(input, output)
            for jj in range(len(self.smList)):
                self.smList[jj].update(input, output)

            if ii%1000 == 0:
                print(f'ii = {ii}')

        self.DMinArrayList = []
        self.TimeArrayList = []
        self.MinDMin = []
        self.MeanDMin = []
        self.StdDMin = []
        self.MeanTime = []
        for ii in range(len(self.rhoList)):
            DMinArray = ( self.smList[ii].DSMin_data ** 0.5 ).ravel()
            self.DMinArrayList.append(DMinArray)
            self.MinDMin.append( DMinArray.min() )
            self.MeanDMin.append( DMinArray.mean() )
            self.StdDMin.append( DMinArray.std() )
            self.TimeArrayList.append( self.smList[ii].time_data.ravel() )
            self.MeanTime.append( self.smList[ii].time_data.mean() )

        self.save_data()

    def save_data(self):
        data = [self.DMinArrayList, self.MinDMin, self.MeanDMin, self.StdDMin, self.TimeArrayList, self.MeanTime, self.smList, self.sample_generation, self.sm]
        with open(self.file_name, 'wb') as f:
            pickle.dump(data, f)
            print('pkl save successfully')

    def read_data(self):
        with open(self.file_name, 'rb') as f:
            data = pickle.load(f)
            print('pkl load successfully')
        self.DMinArrayList, self.MinDMin, self.MeanDMin, self.StdDMin, self.TimeArrayList, self.MeanTime, self.smList, self.sample_generation, self.sm = data

    def result_plot(self):
        self.read_data()
        cmap = 'plasma'
        scatter_size = 1
        dt = 0.01


        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        prop_cycle = plt.rcParams['axes.prop_cycle']

        # 1*3 figure
        labelsize = 15

        Ax = []
        fig = plt.figure(figsize=(15, 4))
        ax = fig.add_subplot(131)
        ax.tick_params(axis='both', labelsize=labelsize)
        X = self.sample_generation.input_loader[:, 0]
        Y = self.sample_generation.input_loader[:, 1]
        C = self.sample_generation.time_loader[:, 0]
        vmin = 1*dt
        vmax = np.max(C.ravel())*dt
        scatter = plt.scatter(X, Y, s=scatter_size, c=C*dt, vmin=vmin, vmax=vmax, cmap=plt.get_cmap(cmap), rasterized=True, alpha=0.9)
        ax.set_xlabel(r'${x}_1$', fontsize=labelsize)
        ax.set_ylabel(r'$u$', fontsize=labelsize)
        plt.title('All samples: ' + str(X.size), fontsize=labelsize)
        XLim = plt.xlim()
        YLim = plt.ylim()
        Ax.append(ax)

        ax = fig.add_subplot(132)
        ax.tick_params(axis='both', labelsize=labelsize)
        plt.scatter(X[-self.smList[0].capacity:], Y[-self.smList[0].capacity:], s=scatter_size,
                    c=C[-self.smList[0].capacity:]*dt,
                    vmin=vmin, vmax=vmax, cmap=plt.get_cmap(cmap), alpha=0.8)
        ax.set_xlabel(r'${x}_1$', fontsize=labelsize)
        ax.set_ylabel(r'$u$', fontsize=labelsize)
        plt.title('Samples extracted by \n sliding window: ' + str(self.smList[0].capacity), fontsize=labelsize)
        plt.xlim(XLim)
        plt.ylim(YLim)
        Ax.append(ax)

        ax = fig.add_subplot(133)
        ax.tick_params(axis='both', labelsize=labelsize)
        X_m = self.sm.input_data[:, 0]
        Y_m = self.sm.input_data[:, 1]
        C_m = self.sm.time_data[:, 0]
        ax.scatter(X_m, Y_m, s=scatter_size, c=C_m*dt, vmin=vmin, vmax=vmax, cmap=plt.get_cmap(cmap))
        ax.set_xlabel(r'${x}_1$', fontsize=labelsize)
        ax.set_ylabel(r'$u$', fontsize=labelsize)
        plt.title('Samples extracted by the\n' + fr'proposed algorithm ($\rho$ = {self.sm.rho}): ' + str(
            self.sm.capacity), fontsize=labelsize)
        plt.xlim(XLim)
        plt.ylim(YLim)
        Ax.append(ax)
        plt.subplots_adjust(left=None, bottom=0.15, right=None, top=None, wspace=0.2, hspace=None)
        cbar = fig.colorbar(scatter, ax=Ax, shrink=0.95, aspect=25, fraction=0.1, pad=0.05)  # pad 颜色条距离左边图形的比例
        cbar.ax.tick_params(labelsize=labelsize)  # 替换10为你希望的字体大小
        plt.text(1.42, 2.3, '$t$ [s]', size=labelsize)
        self.save_show(flag_save=True, flag_show=True, filename='./figure/sm_3sample.pdf', fig=fig, dpi=400)

        # 2*3 figure
        labelsize = 15
        Ax = []
        for ii in range(len(self.rhoList)):
            C_m = self.smList[ii].time_data[:, 0]
            if ii == 0:
                vmin = float(1.0e10)*dt
                vmax = float(-1.0)*dt
            else:
                vmin = np.min(( vmin, C_m.min()*dt ))
                vmax = np.max(( vmax, C_m.max()*dt ))

        fig = plt.figure(figsize=(15, 9))
        for ii in range(len(self.rhoList)):
            ax = fig.add_subplot(2, 3, 1 + ii)
            ax.tick_params(axis='both', labelsize=labelsize)
            X_m = self.smList[ii].input_data[:, 0]
            Y_m = self.smList[ii].input_data[:, 1]
            C_m = self.smList[ii].time_data[:, 0]
            ax.scatter(X_m, Y_m, s=scatter_size, c=C_m*dt, vmin=vmin, vmax=vmax, cmap=plt.get_cmap(cmap))
            ax.set_xlabel(r'${x}_1$', fontsize=labelsize)
            ax.set_ylabel(r'$u$', fontsize=labelsize)
            plt.title(rf'$\rho$ = {self.rhoList[ii]}', fontsize=labelsize)
            plt.xlim(XLim)
            plt.ylim(YLim)
            Ax.append(ax)
        plt.subplots_adjust(left=None, bottom=0.15, right=None, top=None, wspace=0.25, hspace=0.35)
        cbar = fig.colorbar(scatter, ax=Ax, shrink=0.95, aspect=50, fraction=0.1, pad=0.04)  # pad 颜色条距离左边图形的比例
        cbar.ax.tick_params(labelsize=labelsize)  # 替换10为你希望的字体大小
        plt.text(1.35, 8.3, '$t$ [s]', size=labelsize)
        self.save_show(flag_save=True, flag_show=True, filename='./figure/sm_6sample.pdf', fig=fig)

        self.t_and_dmin_plot(dt=0.01)

    def t_and_dmin_plot(self, dt):
        labelsize = 12
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(211)
        ax.tick_params(axis='both', labelsize=labelsize)
        TimeArray = [ t*dt for t in self.TimeArrayList]
        plt.boxplot(TimeArray, labels=self.rhoList, widths=0.2, showfliers=False,
                    medianprops={'linewidth': 1.4, 'color': '#4649CA'})
        plt.ylabel('$t$ [s]', fontsize=labelsize)
        plt.grid(True)
        ax = fig.add_subplot(212)
        ax.tick_params(axis='both', labelsize=labelsize)
        plt.boxplot(self.DMinArrayList, labels=self.rhoList, widths=0.2, showfliers=False,
                    medianprops={'linewidth': 1.4, 'color': '#F2A316'})
        plt.ylabel('$d^{\mathrm{min}}$', fontsize=labelsize)
        plt.grid(True)
        plt.xlabel(r'$\rho$', fontsize=labelsize)
        plt.subplots_adjust(left=0.15, bottom=0.15, right=None, top=None, wspace=0.2, hspace=0.12)
        self.save_show(flag_save=True, flag_show=True, filename='./figure/Boxplot.pdf', fig=fig)

    def save_show(self, flag_save, flag_show, filename, fig, dpi=None):
        if flag_save:
            plt.savefig(filename, bbox_inches='tight', dpi=dpi)
        if not flag_show:
            plt.close(fig)

if __name__ == '__main__':
    test = Test()
    # test.run()
    test.result_plot()
    print('finish!')
    plt.show()

