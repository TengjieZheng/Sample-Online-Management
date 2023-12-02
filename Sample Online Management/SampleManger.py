import numpy as np

"""
Sample Online Management Algorithm 
Author: Tengjie Zheng
Date: [2023.12.1]
Description: [extracted uniform sample set from data stream]
"""


class SampleManager():
    def __init__(self, capacity=100, input_dimension=1, output_dimension=1, rho=0.1):
        """
        capacity:           The upper limit for the number of samples in the sample set [int,>2].
        input_dimension:    The dimension of the input data [int,>0].
        output_dimension:   The dimension of the output data [int,>0].
        rho:                Timeliness factor [float,>=0]. A larger timeliness factor weakens timeliness and strengthens uniformity.
        """

        if capacity <= 2:
            raise TypeError('capacity must > 2')

        # Manager properties
        self.capacity = capacity            # Capacity of the sample set
        self.dim_input = input_dimension    # Dimension of input data
        self.dim_output = output_dimension  # Dimension of output data
        self.num_data = 0                   # Number of samples in the set
        self.time = 0                       # Time
        self.rho = rho                      # Timeliness factor, ranging from 0 to 1. A value of 0 implies maximum timeliness, while larger values indicate weaker timeliness. Generally, it ranges from 0 to 1; a value greater than 1 suggests timeliness correction.

        # Properties of samples in the pool
        self.input_data = np.empty((0, input_dimension))    # Input data
        self.output_data = np.empty((0, output_dimension))  # Output data
        self.DSMin_data = np.empty((0, 1))                  # Minimum DS (Distance Squared) with other samples
        self.time_data = np.empty((0, 1))                   # Sample time of data
        self.IdxNearest_data = np.empty((0, 1))             # ID of the nearest sample

    def update(self, In, Out):
        In = np.array(In).reshape((1, -1))
        Out = np.array(Out).reshape((1, -1))
        self.time += 1
        self.DSArray = np.sum((self.input_data - In) ** 2, axis=1).reshape((-1, 1))

        if self.num_data < self.capacity:
            self.UpdateSampleSet(In, Out)
        else:
            IdxWorst = self.FindWorstData(In, Out)
            if IdxWorst == self.num_data:
                IdxNearEarly = self.FindNearestEarliestData(In, Out)
                T = self.time_data[-1, 0] - self.time_data[0, 0]
                if self.time - self.time_data[IdxNearEarly, 0] > T * self.rho:
                    if IdxNearEarly < self.num_data-1 :
                        self.UpdateSampleSet(In, Out, IdxNearEarly, flag='add and remove')
            else:
                IdxNearEarly = self.FindNearestEarliestData(In, Out)
                IsObviousGood = self.IsObviousGood(In, Out, IdxNearEarly, IdxWorst)
                if IsObviousGood:
                    # 新样本明显重要
                    self.UpdateSampleSet(In, Out, IdxWorst, flag='add and remove')

    def FindWorstData(self, In, Out):
        # Find the worst sample
        InOldNew = np.vstack((self.input_data, In))
        DSMinOldNew, NearestOldNew = self.AddDataUpdateDSminNearest(In, Out, self.DSMin_data, self.IdxNearest_data)
        DSMin = np.min(DSMinOldNew.ravel())
        IdxSeq = np.arange(self.num_data + 1)
        IdxMin = ( IdxSeq[ DSMinOldNew.ravel() == DSMin ] ).ravel()
        if IdxMin.size == 1:
            IdxWorst = IdxMin.ravel()[0]
        else:
            DSSecMin = np.zeros(IdxMin.size)
            for ii, Idx in enumerate(IdxMin):
                InNoMin = np.delete(InOldNew, [ Idx, NearestOldNew.ravel()[Idx] ], axis=0)
                Inii = InOldNew[Idx, :].reshape((1, -1))
                DSNoMinArray = np.sum((InNoMin - Inii) ** 2, axis=1).ravel()
                DSSecMin[ii] = np.min(DSNoMinArray)
            MinDSSecMin = np.min(DSSecMin)
            IdxSecMin = IdxMin[DSSecMin == MinDSSecMin]
            if IdxSecMin.size == 1:
                IdxWorst = IdxSecMin.ravel()[0]
            else:
                TimeNewOld = np.vstack((self.time_data, self.time))
                TimeDSMin = TimeNewOld.ravel()[IdxSecMin]
                IdxWorst = IdxSecMin[np.argmin(TimeDSMin)]

        return IdxWorst

    def FindNearestEarliestData(self, In, Out):
        # Find the data closest and earliest to the new sample
        DSMin = np.min(self.DSArray.ravel())
        IdxSeq = np.arange(self.num_data)
        IdxMin = IdxSeq[self.DSArray.ravel() == DSMin]
        if IdxMin.size == 1:
            IdxNearEarly = IdxMin.ravel()[0]
        else:
            TimeDSMin = self.time_data.ravel()[IdxMin]
            IdxNearEarly = IdxMin[ np.argmin(TimeDSMin) ]

        return IdxNearEarly

    def IsObviousGood(self, In, Out, IdxCheck, IdxMin):
        # Evaluate whether the new sample is obviously good
        InCheck = self.input_data[IdxCheck, :].reshape((1, -1))
        DSNewCheck = np.sum( ( (In - InCheck).ravel() )**2 )
        DSMin = self.DSMin_data[IdxMin, 0]

        if DSNewCheck >= DSMin * 1:
            IsGood = True
        else:
            IsGood = False

        return IsGood

    def UpdateSampleSet(self, In, Out, IdxRemove=None, flag='add'):
        # Update the sample set:
        # When the flag is set to 'add' only new data is added;
        # when the flag is set to 'add and remove,' both new data is added and specified old data is removed.
        if flag == 'add':
            self.AddData(In, Out)
        elif flag == 'add and remove':
            self.RemoveData(IdxRemove)
            self.AddData(In, Out)

    def AddData(self, In, Out):
        # add new data and update the properties of samples in the set
        if self.num_data == 0:
            self.IdxNearest_data = np.vstack((self.IdxNearest_data, 0))
            self.DSMin_data = np.vstack((self.DSMin_data, 0.0))
            self.input_data = np.vstack((self.input_data, In))
            self.output_data = np.vstack((self.output_data, Out))
            self.time_data = np.vstack((self.time_data, self.time))
        elif self.num_data == 1:
            InOld = self.input_data
            DSOldNew = np.sum( ( (InOld - In)**2 ).ravel() )
            self.IdxNearest_data = np.array([[1, 0]]).T
            self.DSMin_data = np.array([[ DSOldNew, DSOldNew ]]).T
            self.input_data = np.vstack((self.input_data, In))
            self.output_data = np.vstack((self.output_data, Out))
            self.time_data = np.vstack((self.time_data, self.time))
        else:
            self.DSMin_data, self.IdxNearest_data = self.AddDataUpdateDSminNearest(In, Out, self.DSMin_data, self.IdxNearest_data)
            self.input_data = np.vstack((self.input_data, In))
            self.output_data = np.vstack((self.output_data, Out))
            self.time_data = np.vstack((self.time_data, self.time))

        self.num_data += 1

    def AddDataUpdateDSminNearest(self, In, Out, DSMin_data, IdxNearest_data):
        # Calculate DSMin_data and IdxNearest_data after adding new data into the sample set
        DSMin_data = np.copy(DSMin_data)
        IdxNearest_data = np.copy(IdxNearest_data)
        DSArray = np.sum((self.input_data - In) ** 2, axis=1).reshape((-1, 1))
        IdxLess = (DSArray < DSMin_data).ravel()
        DSMin_data[IdxLess, :] = DSArray[IdxLess, :]
        IdxNearest_data[IdxLess, :] = DSMin_data.size

        IdxMin = np.argmin(DSArray.ravel())
        IdxNearest_data = np.vstack((IdxNearest_data, IdxMin))
        DSMin_data = np.vstack((DSMin_data, DSArray.ravel()[IdxMin]))

        return DSMin_data, IdxNearest_data

    def RemoveData(self, IdxRemove):
        # remove specified old data and update the properties of samples in the set
        IdxSeq = np.arange(self.num_data)
        IdxChange = IdxSeq[ self.IdxNearest_data.ravel() == IdxRemove ]
        for Idx in IdxChange:
            InNoRemove = np.delete(self.input_data, [Idx, IdxRemove], axis=0)
            IdxNoRemove = np.delete(IdxSeq, [Idx, IdxRemove], axis=0)
            DSNoRemoveArray = np.sum( ( InNoRemove - self.input_data[Idx, :].reshape((1, -1)) )**2, axis=1)
            self.DSMin_data[Idx, :] = np.min(DSNoRemoveArray.ravel())
            self.IdxNearest_data[Idx, :] = IdxNoRemove[ np.argmin(DSNoRemoveArray) ]

        self.time_data = np.delete(self.time_data, IdxRemove, axis=0)
        self.input_data = np.delete(self.input_data, IdxRemove, axis=0)
        self.output_data = np.delete(self.output_data, IdxRemove, axis=0)
        self.DSMin_data = np.delete(self.DSMin_data, IdxRemove, axis=0)
        self.IdxNearest_data = np.delete(self.IdxNearest_data, IdxRemove, axis=0)
        self.IdxNearest_data[(self.IdxNearest_data > IdxRemove).ravel(), 0] -= 1

        self.num_data -= 1

