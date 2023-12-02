import numpy as np
from Object import Object

"""
generate the test data for sample online management algorithm
"""

class Sample_generation():
    def __init__(self, input_dimension=1, sigma_noise=1e-2, flag_record=True, type=0):
        np.random.seed(10)
        self.dim_input = input_dimension
        self.type = type # Data Types: 0 Gaussian Distribution Data, 1 Continuous Data, 2 First-Order System Data
        self.sigma_noise = sigma_noise

        # Parameters for Gaussian Distribution Data
        self.center = 0
        self.sigma = 1 / 3 * 1.3
        self.weight = np.ones((1, self.dim_input))

        # Parameters for Continuous Data
        self.t = 0.0
        self.k = 5  # Relative rate of continuous data generation

        # Parameters for First-Order System Data
        self.obj = Object(delta_t=0.01)

        # Counting and Data Storage
        self.num_samples = 0
        self.input_loader = np.empty((0, input_dimension))
        self.output_loader = np.empty((0, 1))
        self.time_loader = np.empty((0, 1))
        self.flag_data_record = flag_record

    def new_sample(self):

        if self.type == 0:
            mean = self.center * np.ones((1, self.dim_input)).ravel()
            cov = self.sigma ** 2 * np.eye(mean.size)
            input = 2 * np.ones_like(mean)
            while (np.abs(input)>1).sum() > 0:
                input = np.random.multivariate_normal(mean, cov)

            input = self.weight * input
            output = np.ones(1)
            output += np.random.normal(0, scale=self.sigma_noise, size=output.shape)

        elif self.type == 1:
            k = 1
            omega0 = 2 * np.pi / 11400 * self.k
            omega = 2 * np.pi / 2000 * self.k
            input = np.zeros((1, self.dim_input))
            for ii in range(input.size):
                if ii % 2 == 0:
                    x = k * np.sin(self.t * omega0) * np.cos(self.t * omega)
                else:
                    x = k * np.sin(self.t * omega0) * np.sin(self.t * omega)
                input[0 ,ii] = x

            output = np.ones(1)
            output += np.random.normal(0, scale=self.sigma_noise, size=output.shape)
            self.t += 1

        elif self.type == 2:
            self.obj.update()
            input = np.array([self.obj.x, self.obj.u]).reshape((1, -1))
            output = self.obj.f
            output += np.random.normal(0, scale=self.sigma_noise, size=output.shape)

        self.num_samples += 1
        if self.flag_data_record:
            self.input_loader = np.vstack((self.input_loader, input))
            self.output_loader = np.vstack((self.output_loader, output))
            self.time_loader = np.vstack((self.time_loader, self.num_samples))

        return input, output