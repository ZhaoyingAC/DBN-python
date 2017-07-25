import re
import xml.etree.cElementTree as ET
import numpy


################################################################################

def readDataBlock(xmlnode):
    """ Turn any 'DataBlock' XML node into a numpy array of floats
    """
    vmin = float(xmlnode.get('min'))
    vmax = float(xmlnode.get('max'))
    string = xmlnode.text
    string = re.sub("[\t\s\n]", "", string)
    data = numpy.asarray(
        bytearray.fromhex(string),
        dtype=float
    )
    return data * (vmax - vmin) / 255. + vmin


class Candidate(object):
    def __init__(self, fname):
        """ Build a new Candidate object from a PHCX file path.
        """
        xmlroot = ET.parse(fname).getroot()

        # Read Coordinates
        coordNode = xmlroot.find('head').find('Coordinate')
        self.rajd = float(coordNode.find('RA').text)
        self.decjd = float(coordNode.find('Dec').text)

        # Separate PDMP & FFT sections
        for section in xmlroot.findall('Section'):
            if 'pdmp' in section.get('name').lower():
                opt_section = section
            else:
                fft_section = section

        # Best values as returned by PDMP
        opt_values = {
            node.tag: float(node.text)
            for node in opt_section.find('BestValues').getchildren()
            }

        self.bary_period = opt_values['BaryPeriod']
        self.topo_period = opt_values['TopoPeriod']
        self.dm = opt_values['Dm']
        self.snr = opt_values['Snr']
        self.width = opt_values['Width']

        ##### P-DM plane #####
        pdmNode = opt_section.find('SnrBlock')
        # DmIndex
        string = pdmNode.find('DmIndex').text
        #        dm_index = numpy.asarray(map(float, string.split()))
        dm_index = []
        dm_index_str = string.split()
        for dm_index_str_element in dm_index_str:
            dm_index_element = float(dm_index_str_element)
            dm_index.append(dm_index_element)
        dm_index = numpy.asarray(dm_index)
        #        print (dm_index.size)
        # PeriodIndex
        string = pdmNode.find('PeriodIndex').text

        #        period_index = numpy.asarray(map(float, string.split()))
        period_index_str = string.split()
        period_index = []
        for period_index_str_element in period_index_str:
            period_index_element = float(period_index_str_element)
            period_index_element = period_index_element / 1.0e12
            period_index.append(period_index_element)
        period_index = numpy.asarray(period_index)
        #        print (period_index.size)
        #        period_index /= 1.0e12 # Picoseconds to seconds

        # S/N data
        pdmDataNode = pdmNode.find('DataBlock')
        #        print (readDataBlock(pdmDataNode).size)
        #        print (dm_index)
        pdm_plane = readDataBlock(pdmDataNode).reshape(
            dm_index.size,
            period_index.size
        )

        # Pack all P-DM plane arrays into a tuple
        self.pdm_plane = (period_index, dm_index, pdm_plane)

        ### Sub-Integrations
        subintsNode = opt_section.find('SubIntegrations')
        nsubs = int(subintsNode.get('nSub'))
        nbins = int(subintsNode.get('nBins'))
        self.subints = readDataBlock(subintsNode).reshape(nsubs, nbins)

        ### Sub-Bands
        subbandsNode = opt_section.find('SubBands')
        nsubs = int(subbandsNode.get('nSub'))
        nbins = int(subbandsNode.get('nBins'))
        self.subbands = readDataBlock(subbandsNode).reshape(nsubs, nbins)

        ### Profile
        profileNode = opt_section.find('Profile')
        self.profile = readDataBlock(profileNode)

        ##### Parse FFT Section (PEASOUP Data) #####
        fft_values = {
            node.tag: float(node.text)
            for node in fft_section.find('BestValues').getchildren()
            }
        self.accn = fft_values['Accn']
        self.hits = fft_values['Hits']
        self.rank = fft_values['Rank']
        self.fftsnr = fft_values['SpectralSnr']

        ### DmCurve: FFT S/N vs. PEASOUP Trial DM, at best candidate acceleration
        dmcurve_node = fft_section.find('DmCurve')

        text = dmcurve_node.find('DmValues').text
        dm_values = numpy.asarray([float(value) for value in text.split()])
        # dm_values = numpy.asarray(map(float, text.split()))

        text = dmcurve_node.find('SnrValues').text
        snr_values = numpy.asarray([float(value) for value in text.split()])
        # snr_values = numpy.asarray(map(float, text.split()))

        # Pack the DM curve into a tuple of arrays
        self.dm_curve = (dm_values, snr_values)

        ### AccnCurve: FFT S/N vs. PEASOUP Trial Acc, at best candidate DM
        accncurve_node = fft_section.find('AccnCurve')

        text = accncurve_node.find('AccnValues').text
        # accn_values = numpy.asarray(map(float, text.split()))
        accn_values = numpy.asarray([float(value) for value in text.split()])
        text = accncurve_node.find('SnrValues').text
        snr_values = numpy.asarray([float(value) for value in text.split()])
        # snr_values = numpy.asarray(map(float, text.split()))

        # Pack the Accn curve into a tuple of arrays
        self.accn_curve = (accn_values, snr_values)


################################################################################

def phase_plots(cand, desFile):
    import pylab as plt
    plt.figure(1, figsize=(9, 7), dpi=70)
    plt.subplot(311)
    plt.imshow(cand.subbands, origin='lower', interpolation='nearest', aspect='auto', cmap=plt.cm.Greys)
    plt.title('Sub-Bands')
    plt.ylabel('Band Index')

    plt.subplot(312)
    plt.imshow(cand.subints, origin='lower', interpolation='nearest', aspect='auto', cmap=plt.cm.Greys)
    plt.title('Sub-Integrations')
    plt.ylabel('Integration Index')

    plt.subplot(313)
    plt.bar(range(cand.profile.size), cand.profile, width=1)
    plt.xlim(0, cand.profile.size)
    plt.xlabel('Phase Bin Index')

    plt.tight_layout()
    #   plt.show()
    plt.savefig(desFile, format='png')
    plt.close()


def bullseye_plot(cand, desFile):
    import pylab as plt
    p, dm, snr = cand.pdm_plane

    plt.figure(2, figsize=(7, 5), dpi=80)
    # IMPORTANT NOTE: imshow() must be called with origin='lower' here, otherwise
    # the DM values on the Y axis are reversed (and therefore wrong).
    plt.imshow(
        snr,
        extent=[p.min(), p.max(), dm.min(), dm.max()],
        aspect='auto',
        origin='lower',
        interpolation='nearest'
    )
    plt.xlabel('Period Correction (s)')
    plt.ylabel('Trial DM')

    cb = plt.colorbar()
    cb.set_label('Folded S/N')

    plt.tight_layout()
    #    plt.show()


    plt.savefig(desFile, format='png')
    plt.close()


################################################################################

if __name__ == '__main__':
    import os
    import numpy as np
    import csv

    # Load example.phcx file (must be in the same directory as this python script)
    '''
    directory, fname = os.path.split(
        os.path.abspath(__file__)
        )
    cand = Candidate(
        os.path.join(directory, 'example.phcx')
        )
    '''

    candidate_path_base = 'G:\\暑期集训\\MedlatTrainingData\\'
    max = 0
    for child_path in os.listdir(candidate_path_base):
        if "negatives_" in child_path and "tar.gz" not in child_path:
            candidate_path = os.path.join(candidate_path_base, child_path)

            for file_path in os.listdir(candidate_path):
                path = os.path.join(candidate_path, file_path)
                cand = Candidate(path)
                attrs = list()
                sub_bands = cand.subbands.flatten().tolist()
                attrs.extend(sub_bands)
                sub_ins = cand.subints.flatten().tolist()
                attrs.extend(sub_ins)
                # sub_bands.append(1)
                profile = cand.profile              # 1-d [64, ]
                attrs.extend(profile)
                dm = cand.dm                        # a float
                attrs.append(dm)
                dm_curve = cand.dm_curve            # tuple(ndarray, ndarray) (2, 1344)
                attrs.extend(dm_curve[0].tolist())
                attrs.extend(dm_curve[-1].tolist())
                topo = cand.topo_period             # a float
                attrs.append(topo)
                bary = cand.bary_period             # a float
                attrs.append(bary)
                accn = cand.accn                    # a float
                attrs.append(accn)
                accn_curve = cand.accn_curve        # tuple(ndarray, ndarray) (2, 71)
                attrs.extend(accn_curve[0].tolist())
                attrs.extend(accn_curve[-1].tolist())
                attrs.append(0)
                ra = cand.rajd                      # a float
                dec = cand.decjd                    # a float
                num = np.shape(attrs)[0]
                if(num > max):
                    max = num
    print(max)
    # for path in os.listdir(candidate_path_base):
    #     path = candidate_path_base + path
    #     name = path.split('\\')[-1].split('.')[0]
    #     # path = path.replace('\\', '\\\\')
    #     cand = Candidate(path)
    #     attrs = list()
    #     sub_bands = cand.subbands.flatten().tolist()
    #     attrs.extend(sub_bands)
    #     sub_ins = cand.subints.flatten().tolist()
    #     attrs.extend(sub_ins)
    #     # sub_bands.append(1)
    #     profile = cand.profile              # 1-d [64, ]
    #     attrs.extend(profile)
    #     dm = cand.dm                        # a float
    #     attrs.append(dm)
    #     dm_curve = cand.dm_curve            # tuple(ndarray, ndarray) (2, 1344)
    #     attrs.extend(dm_curve[0].tolist())
    #     attrs.extend(dm_curve[-1].tolist())
    #     topo = cand.topo_period             # a float
    #     attrs.append(topo)
    #     bary = cand.bary_period             # a float
    #     attrs.append(bary)
    #     accn = cand.accn                    # a float
    #     attrs.append(accn)
    #     accn_curve = cand.accn_curve        # tuple(ndarray, ndarray) (2, 71)
    #     attrs.extend(accn_curve[0].tolist())
    #     attrs.extend(accn_curve[-1].tolist())
    #     attrs.append(0)
    #     ra = cand.rajd                      # a float
    #     dec = cand.decjd                    # a float
    #     print("Processing: ", name)
    #     print(np.shape(attrs))
        # with open('F:\\PycharmProjects\\DBN-python\\Data\\no_pulsar.csv', 'a', newline='') as file:
        #     writer = csv.writer(file)
        #     writer.writerow(attrs)

    # Make some cool plots
    # phase_plots(cand, 'F:\\PycharmProjects\\DBN-python\\Data\\phase_plot')
    # bullseye_plot(cand, 'F:\\PycharmProjects\\DBN-python\\Data\\phase_plot')
    # print(cand.subints.shape)
    # print(cand.subbands.shape)