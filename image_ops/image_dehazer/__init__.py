# Adapted for 16-bit from https://github.com/Utkarsh-Deshmukh/Single-Image-Dehazing-Python/blob/master/image_dehazer/__init__.py 
# which is based on Efficient Image Dehazing with Boundary Constraint and Contextual Regularization

import cv2
import numpy as np
import copy
import tifffile
import numpy as np
from PIL import Image


def read_image(image_path):
    norm_factor = 255.0  # Default for 8-bit images
    if image_path.endswith(".tif"):
        img = tifffile.imread(image_path)
        
        # Determine normalization factor
        if img.dtype == np.uint8:
            norm_factor = 255.0
        elif img.dtype == np.uint16:
            norm_factor = 65535.0
        else:
            raise ValueError(f"Unsupported image bit depth: {img.dtype}")

        # Normalize to float32 in range [0, 1]
        img = (img.astype(np.float32) / norm_factor)

        # Convert grayscale images to 3-channel
        if img.ndim == 2:  
            img = np.stack([img] * 3, axis=-1)

    else:
        img = Image.open(image_path).convert("RGB")
        img = np.array(img).astype(np.float32) / 255.0  # Normalize to [0,1]

    return img, norm_factor


class image_dehazer():
    def __init__(self, norm_factor=255.0, airlightEstimation_windowSze=15, boundaryConstraint_windowSze=3,
                 C0=20, C1=300, regularize_lambda=0.1, sigma=0.5, delta=0.85, showHazeTransmissionMap=True):
        self.norm_factor = norm_factor  # Store normalization factor (255 or 65535)
        self.airlightEstimation_windowSze = airlightEstimation_windowSze
        self.boundaryConstraint_windowSze = boundaryConstraint_windowSze
        self.C0 = C0 / self.norm_factor  # Normalize thresholds
        self.C1 = C1 / self.norm_factor  # Normalize thresholds
        self.regularize_lambda = regularize_lambda
        self.sigma = sigma
        self.delta = delta
        self.showHazeTransmissionMap = showHazeTransmissionMap
        self._A = []
        self._transmission = []
        self._WFun = []

    def __AirlightEstimation(self, HazeImg):
        if len(HazeImg.shape) == 3:
            for ch in range(HazeImg.shape[2]):
                kernel = np.ones((self.airlightEstimation_windowSze, self.airlightEstimation_windowSze), np.uint8)
                minImg = cv2.erode(HazeImg[:, :, ch], kernel)
                self._A.append(minImg.max())
        else:
            kernel = np.ones((self.airlightEstimation_windowSze, self.airlightEstimation_windowSze), np.uint8)
            minImg = cv2.erode(HazeImg, kernel)
            self._A.append(minImg.max())


    def __BoundCon(self, HazeImg):
        if len(HazeImg.shape) == 3:
            t_b = np.maximum((self._A[0] - HazeImg[:, :, 0]) / (self._A[0] - self.C0),
                             (HazeImg[:, :, 0] - self._A[0]) / (self.C1 - self._A[0]))
            t_g = np.maximum((self._A[1] - HazeImg[:, :, 1]) / (self._A[1] - self.C0),
                             (HazeImg[:, :, 1] - self._A[1]) / (self.C1 - self._A[1]))
            t_r = np.maximum((self._A[2] - HazeImg[:, :, 2]) / (self._A[2] - self.C0),
                             (HazeImg[:, :, 2] - self._A[2]) / (self.C1 - self._A[2]))

            MaxVal = np.maximum(t_b, t_g, t_r)
            self._Transmission = np.minimum(MaxVal, 1.0)
        else:
            self._Transmission = np.maximum((self._A[0] - HazeImg) / (self._A[0] - self.C0),
                                            (HazeImg - self._A[0]) / (self.C1 - self._A[0]))
            self._Transmission = np.minimum(self._Transmission, 1.0)


    def __LoadFilterBank(self):
        KirschFilters = []
        KirschFilters.append(np.array([[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]]))
        KirschFilters.append(np.array([[-3, -3, -3], [-3, 0, -3], [5, 5, 5]]))
        KirschFilters.append(np.array([[-3, -3, -3], [5, 0, -3], [5, 5, -3]]))
        KirschFilters.append(np.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]]))
        KirschFilters.append(np.array([[5, 5, -3], [5, 0, -3], [-3, -3, -3]]))
        KirschFilters.append(np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]]))
        KirschFilters.append(np.array([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]]))
        KirschFilters.append(np.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]]))
        KirschFilters.append(np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]))
        return (KirschFilters)

    def __CalculateWeightingFunction(self, HazeImg, Filter):

        # Computing the weight function... Eq (17) in the paper

        HazeImageDouble = HazeImg.astype(float) / 255.0
        if (len(HazeImg.shape) == 3):
            Red = HazeImageDouble[:, :, 2]
            d_r = self.__circularConvFilt(Red, Filter)

            Green = HazeImageDouble[:, :, 1]
            d_g = self.__circularConvFilt(Green, Filter)

            Blue = HazeImageDouble[:, :, 0]
            d_b = self.__circularConvFilt(Blue, Filter)

            return (np.exp(-((d_r ** 2) + (d_g ** 2) + (d_b ** 2)) / (2 * self.sigma * self.sigma)))
        else:
            d = self.__circularConvFilt(HazeImageDouble, Filter)
            return (np.exp(-((d ** 2) + (d ** 2) + (d ** 2)) / (2 * self.sigma * self.sigma)))

    def __circularConvFilt(self, Img, Filter):
        FilterHeight, FilterWidth = Filter.shape
        assert (FilterHeight == FilterWidth), 'Filter must be square in shape --> Height must be same as width'
        assert (FilterHeight % 2 == 1), 'Filter dimension must be a odd number.'

        filterHalsSize = int((FilterHeight - 1) / 2)
        rows, cols = Img.shape
        PaddedImg = cv2.copyMakeBorder(Img, filterHalsSize, filterHalsSize, filterHalsSize, filterHalsSize,
                                       borderType=cv2.BORDER_WRAP)
        FilteredImg = cv2.filter2D(PaddedImg, -1, Filter)
        Result = FilteredImg[filterHalsSize:rows + filterHalsSize, filterHalsSize:cols + filterHalsSize]
        return (Result)

    def __CalTransmission(self, HazeImg):
        rows, cols = self._Transmission.shape

        KirschFilters = self.__LoadFilterBank()

        # Normalize the filters
        for idx, currentFilter in enumerate(KirschFilters):
            KirschFilters[idx] = KirschFilters[idx] / np.linalg.norm(currentFilter)

        # Calculate Weighting function --> [rows, cols. numFilters] --> One Weighting function for every filter
        WFun = []
        for idx, currentFilter in enumerate(KirschFilters):
            WFun.append(self.__CalculateWeightingFunction(HazeImg, currentFilter))

        # Precompute the constants that are later needed in the optimization step
        tF = np.fft.fft2(self._Transmission)
        DS = 0

        for i in range(len(KirschFilters)):
            D = self.__psf2otf(KirschFilters[i], (rows, cols))
            # D = psf2otf(KirschFilters[i], (rows, cols))
            DS = DS + (abs(D) ** 2)

        # Cyclic loop for refining t and u --> Section III in the paper
        beta = 1  # Start Beta value --> selected from the paper
        beta_max = 2 ** 4  # Selected from the paper --> Section III --> "Scene Transmission Estimation"
        beta_rate = 2 * np.sqrt(2)  # Selected from the paper

        while (beta < beta_max):
            gamma = self.regularize_lambda / beta

            # Fixing t first and solving for u
            DU = 0
            for i in range(len(KirschFilters)):
                dt = self.__circularConvFilt(self._Transmission, KirschFilters[i])
                u = np.maximum((abs(dt) - (WFun[i] / (len(KirschFilters) * beta))), 0) * np.sign(dt)
                DU = DU + np.fft.fft2(self.__circularConvFilt(u, cv2.flip(KirschFilters[i], -1)))

            # Fixing u and solving t --> Equation 26 in the paper
            # Note: In equation 26, the Numerator is the "DU" calculated in the above part of the code
            # In the equation 26, the Denominator is the DS which was computed as a constant in the above code

            self._Transmission = np.abs(np.fft.ifft2((gamma * tF + DU) / (gamma + DS)))
            beta = beta * beta_rate

        # if (self.showHazeTransmissionMap):
        #     cv2.imshow("Haze Transmission Map", self._Transmission)
        #     cv2.waitKey(1)

    def __removeHaze(self, HazeImg):
        epsilon = 0.0001
        Transmission = np.power(np.maximum(abs(self._Transmission), epsilon), self.delta)

        HazeCorrectedImage = np.copy(HazeImg)
        if len(HazeImg.shape) == 3:
            for ch in range(HazeImg.shape[2]):
                temp = ((HazeImg[:, :, ch] - self._A[ch]) / Transmission) + self._A[ch]
                temp = np.clip(temp, 0, 1.0)  # Ensure valid range for 8-bit/16-bit
                HazeCorrectedImage[:, :, ch] = temp
        else:
            temp = ((HazeImg - self._A[0]) / Transmission) + self._A[0]
            temp = np.clip(temp, 0, 1.0)
            HazeCorrectedImage = temp
        return HazeCorrectedImage
    

    def __psf2otf(self, psf, shape):
        '''
            this code is taken from:
            https://pypi.org/project/pypher/
        '''
        """
        Convert point-spread function to optical transfer function.

        Compute the Fast Fourier Transform (FFT) of the point-spread
        function (PSF) array and creates the optical transfer function (OTF)
        array that is not influenced by the PSF off-centering.
        By default, the OTF array is the same size as the PSF array.

        To ensure that the OTF is not altered due to PSF off-centering, PSF2OTF
        post-pads the PSF array (down or to the right) with zeros to match
        dimensions specified in OUTSIZE, then circularly shifts the values of
        the PSF array up (or to the left) until the central pixel reaches (1,1)
        position.

        Parameters
        ----------
        psf : `numpy.ndarray`
            PSF array
        shape : int
            Output shape of the OTF array

        Returns
        -------
        otf : `numpy.ndarray`
            OTF array

        Notes
        -----
        Adapted from MATLAB psf2otf function

        """
        if np.all(psf == 0):
            return np.zeros_like(psf)

        inshape = psf.shape
        # Pad the PSF to outsize
        psf = self.__zero_pad(psf, shape, position='corner')

        # Circularly shift OTF so that the 'center' of the PSF is
        # [0,0] element of the array
        for axis, axis_size in enumerate(inshape):
            psf = np.roll(psf, -int(axis_size / 2), axis=axis)

        # Compute the OTF
        otf = np.fft.fft2(psf)

        # Estimate the rough number of operations involved in the FFT
        # and discard the PSF imaginary part if within roundoff error
        # roundoff error  = machine epsilon = sys.float_info.epsilon
        # or np.finfo().eps
        n_ops = np.sum(psf.size * np.log2(psf.shape))
        otf = np.real_if_close(otf, tol=n_ops)

        return otf

    def __zero_pad(self, image, shape, position='corner'):
        """
        Extends image to a certain size with zeros

        Parameters
        ----------
        image: real 2d `numpy.ndarray`
            Input image
        shape: tuple of int
            Desired output shape of the image
        position : str, optional
            The position of the input image in the output one:
                * 'corner'
                    top-left corner (default)
                * 'center'
                    centered

        Returns
        -------
        padded_img: real `numpy.ndarray`
            The zero-padded image

        """
        shape = np.asarray(shape, dtype=int)
        imshape = np.asarray(image.shape, dtype=int)

        if np.all(imshape == shape):
            return image

        if np.any(shape <= 0):
            raise ValueError("ZERO_PAD: null or negative shape given")

        dshape = shape - imshape
        if np.any(dshape < 0):
            raise ValueError("ZERO_PAD: target size smaller than source one")

        pad_img = np.zeros(shape, dtype=image.dtype)

        idx, idy = np.indices(imshape)

        if position == 'center':
            if np.any(dshape % 2 != 0):
                raise ValueError("ZERO_PAD: source and target shapes "
                                 "have different parity.")
            offx, offy = dshape // 2
        else:
            offx, offy = (0, 0)

        pad_img[idx + offx, idy + offy] = image

        return pad_img

    def remove_haze(self, HazeImg):
        self.__AirlightEstimation(HazeImg)
        self.__BoundCon(HazeImg)
        self.__CalTransmission(HazeImg)
        haze_corrected_img = self.__removeHaze(HazeImg)
        HazeTransmissionMap = self._Transmission
        return (haze_corrected_img, HazeTransmissionMap)


def remove_haze(img, norm_factor, airlightEstimation_windowSze=15,
                boundaryConstraint_windowSze=3, C0=20, C1=300,
                regularize_lambda=0.1, sigma=0.5, delta=0.85, showHazeTransmissionMap=False):
    
    # HazeImg, norm_factor = read_image(img_path)

    Dehazer = image_dehazer(norm_factor=norm_factor,
                            airlightEstimation_windowSze=airlightEstimation_windowSze,
                            boundaryConstraint_windowSze=boundaryConstraint_windowSze, 
                            C0=C0, C1=C1,
                            regularize_lambda=regularize_lambda, sigma=sigma, delta=delta,
                            showHazeTransmissionMap=showHazeTransmissionMap)
    
    HazeCorrectedImg, HazeTransmissionMap = Dehazer.remove_haze(img)
    return HazeCorrectedImg, HazeTransmissionMap
