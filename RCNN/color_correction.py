import numpy as np
import math
import glob
import os
import cv2
import matplotlib.pyplot as plt

def color_correction(pixels, x = 0, y = 0, adjustment_intensity = 1, _filter = None):
    """
    This function takes in an image and color correct it based on the average red colors in it.
    It support customary blue value adjustment, adjustment intensity, and indices to make the result
    more or less red/violet. The default intensity is 1, and there is no green or violet adjusment at default.
    :pixels: the image, comes in with RGB color space
    :x: the index to make the result more violet, in range of -1 to 1
    :y: the index to make the result more red, in range of -1 to 1
    :adjustment_intensity: scale the intensity of the adjustment from the identity matrix, in range of [0,+)
    :_filter: the customary matrix that the user may supply
    """
#     pixels = cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)
    if _filter is None:
        _filter = getColorFilterMatrix(pixels, x, y, adjustment_intensity = adjustment_intensity)
    transform_matrix = np.transpose(np.array([
        [_filter[0][0], _filter[0][1], _filter[0][2]],
        [0, _filter[1][1], 0],
        [0, 0, _filter[2][2]]
    ]))
    result = np.matmul(pixels, transform_matrix)
    result[:,:,0] += _filter[0][4] * 255
    result[:,:,1] += _filter[1][4] * 255
    result[:,:,2] += _filter[2][4] * 255
    result = np.where(result < 0, 0, result)
    result = np.where(result > 255, 255, result)
    result = np.float32(result)
    return result.astype(int)

def getColorFilterMatrix(pixels, x, y, adjustment_intensity = 1):
    """
    Compute the color filter matrix of the given img
    :pixels: the three dimensional array of the img pixels, with G,B,R sequence
    :blue_val: the specifed blue value for the image to be correctioned
    """
    # Magic values:
    numOfPixels = pixels.shape[0] * pixels.shape[1]
    thresholdRatio = 2000
    thresholdLevel = numOfPixels / thresholdRatio
    minAvgRed = 60
    maxHueShift = 120
    blueMagicValue = 1.2 # 0.3

    # Objects:
    normalize = {"r":[], "g":[], "b":[]}
    adjust = {"r":[], "g":[], "b":[]}
    hueShift = 0
    
    # calculate the average RGB color of the image
    avg = calculateAverageColor(pixels)

    # Calculate shift amount:
    newAvgRed = avg[0][0][0]
    while newAvgRed < minAvgRed:
        shifted = hueShiftRed(avg, hueShift)
        newAvgRed = shifted.sum()
        hueShift += 1
        if hueShift > maxHueShift: newAvgRed = 60 # Max value
    
    # Create histogram with new red values:
    rounded = np.round(pixels)
    shifted = hueShiftRed(rounded, 60)
    
    # shifted red is summed using the hew shifted RGB values
    shifted_red = shifted.sum(axis = 2)
    
    # shifted red should be between 0 and 255
    shifted_red = np.where(shifted_red < 0, 0, shifted_red)
    shifted_red = np.where(shifted_red > 255, 255, shifted_red)
    
    # round the newly computed shifted red
    shifted_red = np.round(shifted_red).astype(int)
    rounded[:,:,0] = shifted_red
    
    # insert a 0 for the bincount to work
    rounded = np.insert(rounded, rounded.shape[1], 255, axis = 1)
    rounded = np.insert(rounded, 0, 0, axis = 1)
    
    # compute the threshold for the hist
    r_threshold = (np.bincount(rounded[:,:,0].flatten()) - thresholdLevel) < 2
    g_threshold = (np.bincount(rounded[:,:,1].flatten()) - thresholdLevel) < 2
    b_threshold = (np.bincount(rounded[:,:,2].flatten()) - thresholdLevel) < 2
            
    # Push 0 as start value in normalize array:
    normalize['r'].append(0)
    normalize['g'].append(0)
    normalize['b'].append(0)

    # Find values under threshold:
    seq = np.arange(256)

    normalize["r"] += list(seq[r_threshold])
    normalize["g"] += list(seq[g_threshold])
    normalize["b"] += list(seq[b_threshold])
            
    # Push 255 as end value in normalize array:
    normalize["r"].append(255)
    normalize["g"].append(255)
    normalize["b"].append(255)
    
    # compute the adjusted pixels
    adjust["r"] = normalizingInterval(normalize['r'])
    adjust["g"] = normalizingInterval(normalize['g'])
    adjust["b"] = normalizingInterval(normalize['b'])

    # Make histogram:
    shifted = hueShiftRed([1,1,1], hueShift)

    redGain = 256 / (adjust['r']['high'] - adjust['r']['low'])
    greenGain = 256 / (adjust['g']['high'] - adjust['g']['low'])
    blueGain = 256 / (adjust['b']['high'] - adjust['b']['low'])

    redOffset = (-adjust['r']['low'] / 256) * redGain
    greenOffset = (-adjust['g']['low'] / 256) * greenGain
    blueOffset = (-adjust['b']['low'] / 256) * blueGain
    
    adjstRed = shifted[0] * redGain
    adjstRedGreen = shifted[1] * redGain
    adjstRedBlue = shifted[2] * redGain * blueMagicValue
    res =  np.array([
        [adjstRed, adjstRedGreen, adjstRedBlue, 0, redOffset],
        [0, greenGain, 0, 0, greenOffset],
        [0, 0, blueGain, 0, blueOffset],
        [0, 0, 0, 1, 0]
    ])
    
    nominal_mat = np.array([
        [1,0,0,0,0],
        [0,1,0,0,0],
        [0,0,1,0,0],
        [0,0,0,1,0]
    ])
    
    res =  nominal_mat + adjustment_intensity * (res - nominal_mat)
    
    # adjust x
    if x > 0:
        res += x * np.array([
            [0,0,0,0,0],
            [0,0.208,0,0,-0.271],
            [0,0,0.063,0,0],
            [0,0,0,0,0]
        ])
    else:
        res += x * np.array([
            [0,0,0,0,0],
            [0,-0.063,0,0,0],
            [0,0,-0.208,0,0.271],
            [0,0,0,0,0]
        ])
    # adjust y
    res[0] += y * np.array([-0.2, -0.175, -0.125, 0, 0.325])
    return res
    

def calculateAverageColor(pixels):
    """
    :pixels: an 3D RGB image with (height * width * 3) shape
    :return: an 3D array with (1 * 1 * 3) shape containing the average information of the image
    """
    return [[np.average(pixels, axis = (0,1))]]

def hueShiftRed(rgb, h):
    """
    :rgb: a (height * width * 3) shaped 3D numpy array 
    :return: (height * width * 3) shaped 3D numpy array with the RGB value with hue shifted
    """
    U = math.cos(h * math.pi / 180)
    W = math.sin(h * math.pi / 180)
    
    # calculate the transform matrix
    transform_matrix = np.transpose(np.array([
        [(0.299 + 0.701 * U + 0.168 * W), 0, 0],
        [0, (0.587 - 0.587 * U + 0.330 * W), 0],
        [0, 0, (0.114 - 0.114 * U - 0.497 * W)]
    ]))
    return np.matmul(rgb, transform_matrix)

def normalizingInterval(normArray):
    high = 255
    low = 0
    maxDist = 0

    for i in range(1, len(normArray)):
        dist = normArray[i] - normArray[i - 1]
        if dist > maxDist:
            maxDist = dist
            high = normArray[i]
            low = normArray[i - 1]

    return { "low": low, "high": high }