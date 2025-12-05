import pysam
import operator
import numpy as np
from time import time, sleep
import gc
import os

import snvs.constants as c
from snvs.generate_training_data import get_ref_base, get_reads, is_usable_read

def has_hard_or_soft_clips_in_the_middle(read):
    cigar = read.cigarstring # e.g. 1S2D3M
    cigar_letters = [] # e.g. ['S', 'D', 'M']
    
    for char in cigar:
        if char.isalpha():
            cigar_letters.append(char)

    for idx, letter in cigar_letters:
        if (letter == 'S' or letter == 'H') and not (idx == 0 or idx == (len(cigar_letters) - 1) ):
            return True

    return False

def fully_hard_or_soft_clipped(read):
    cigar = read.cigarstring # e.g. 1S2D3M
    cigar_letters = [] # e.g. ['S', 'D', 'M']
    
    for char in cigar:
        if char.isalpha():
            cigar_letters.append(char)

    has_non_soft_or_hard_clip_bases = False
    for letter in cigar_letters:
        if letter != 'S' and letter != 'H':
            has_non_soft_or_hard_clip_bases = True

    return not has_non_soft_or_hard_clip_bases

def check_read(read):
    if read.is_unmapped or read.is_duplicate:
        return False

    if read.mapping_quality < c.MIN_READ_MAPPING_QUALITY:
        return False

    # inspired by BadCigarFilter in GATK: https://software.broadinstitute.org/gatk/documentation/tooldocs/current/org_broadinstitute_gatk_engine_filters_BadCigarFilter.php
    # if fully_hard_or_soft_clipped(read):
    #     return False

    # if has_hard_or_soft_clips_in_the_middle(read):
    #     return False

    return True

def mad(arr):
    """ Median Absolute Deviation: a "Robust" version of standard deviation.
        Indices variabililty of the sample.
        https://en.wikipedia.org/wiki/Median_absolute_deviation 
    """
    arr = np.asarray(arr)
    med = np.median(arr)
    return np.median(np.abs(arr - med))

def goldset_pre_filters(sample_name, bamname_n, bamname_t, chrom, start, end, batch_num):
    print(("\n----- Running region: CHROMOSOME %s, %s-%s ------" % (str(chrom), str(start), str(end))))
    
    batch_filename = os.path.join(c.filtering_folder, c.filtering_batches_folder, sample_name, 'chrom%s_batch_%s.csv' % (chrom, str(batch_num)))

    if os.path.exists(batch_filename):
        print(("Skipping %s as it exists" % batch_filename))
        return

    bamfile_n = pysam.AlignmentFile(bamname_n, 'rb')
    bamfile_t = pysam.AlignmentFile(bamname_t, 'rb')

    print("Calculating coverage on normal and tumor...")
    coverage_n = bamfile_n.count_coverage(chrom, start, end + 1)
    coverage_t = bamfile_t.count_coverage(chrom, start, end + 1)

    print("Pre-processing coverage... ")
    # [ (#A, #C, #G, #T), (#A, #C, #G, #T), (#A, #C, #G, #T), ] at each position in normal
    coverage_list_n = [(coverage_n[0][i], coverage_n[1][i], coverage_n[2][i], coverage_n[3][i]) 
        for i in range(len(coverage_n[0]))]
    del coverage_n

    # [ (#A, #C, #G, #T), (#A, #C, #G, #T), (#A, #C, #G, #T), ] at each position in tumor
    coverage_list_t = [(coverage_t[0][i], coverage_t[1][i], coverage_t[2][i], coverage_t[3][i]) 
        for i in range(len(coverage_t[0]))]
    
    del coverage_t

    print("Starting to filter...")
    positions = []

    for pos in range(start, end + 1):
        tumor_reads = bamfile_t.fetch(chrom, pos, pos + 1)
        normal_reads = bamfile_n.fetch(chrom, pos, pos + 1)

        shortest_distances_from_position_to_alignment_end_in_tumor = []
        base_qualities_in_tumor, read_mapping_qualities_in_normal, read_mapping_qualities_in_tumor = [], [], []
        tumor_read_count, reverse_strand_reads, LOW_MAP_QUAL_READS = 0, 0, 0

        for read in normal_reads:
            read_mapping_qualities_in_normal.append(read.mapping_quality)

        for read in tumor_reads:
            if read.is_reverse:
                reverse_strand_reads += 1

            tumor_read_count += 1

            read_mapping_qualities_in_tumor.append(read.mapping_quality)

            if read.mapping_quality < 1:
                LOW_MAP_QUAL_READS += 1

            distance_from_start, distance_from_end = None, None
            
            if read.reference_start:
                distance_from_start = pos - read.reference_start

            if read.reference_end: 
                distance_from_end = (read.reference_end - 1) - pos # reference_end points to one past the last aligned residue. Returns None if not available (read is unmapped or no cigar alignment present

            if distance_from_start and distance_from_end:
                shortest_distances_from_position_to_alignment_end_in_tumor.append(min(distance_from_start, distance_from_end))
            elif distance_from_start:
                shortest_distances_from_position_to_alignment_end_in_tumor.append(distance_from_start)
            elif distance_from_end:
                shortest_distances_from_position_to_alignment_end_in_tumor.append(distance_from_end)
            else:
                continue

            # get base qualities at pos
            for p in read.get_aligned_pairs():
                if p[1] == pos and p[0] is not None: # p[0] implies Deletion
                    base_qualities_in_tumor.append(read.query_qualities[p[0]])

        CANDIDATE_ALLELE_PRESENT = False
        for allele, allele_count in enumerate(coverage_list_t[pos-start]):
            if allele_count >= c.MIN_VARIANT_ALLELE_COUNT and coverage_list_n[pos-start][allele] <= c.MAX_VARIANT_ALLELE_COUNT_IN_CONTROL:
                CANDIDATE_ALLELE_PRESENT = True

        ALLELE__FREQ_FILTER = not CANDIDATE_ALLELE_PRESENT

        STRAND_BIAS_FILTER, LOW_MAP_QUAL_READS_FILTER = False, False
        if tumor_read_count:
            STRAND_BIAS_FILTER = ((float(reverse_strand_reads)/float(tumor_read_count)) < c.MIN_STRAND_BIAS) or ((float(tumor_read_count-reverse_strand_reads)/float(tumor_read_count)) < c.MIN_STRAND_BIAS)
            LOW_MAP_QUAL_READS_FILTER = (float(LOW_MAP_QUAL_READS)/float(tumor_read_count)) > c.MAX_PROPORTION_OF_LOW_MAP_QUAL_READS_AT_VARIANT
 
        MEDIAN_DISTANCE_TO_END_FILTER = np.median(shortest_distances_from_position_to_alignment_end_in_tumor) < c.MIN_DISTANCE_FROM_VARIANT_TO_ALIGNMENT_END_MEDIAN 
        MAD_DISTANCE_TO_END_FILTER = mad(shortest_distances_from_position_to_alignment_end_in_tumor) < c.MIN_DISTANCE_FROM_VARIANT_TO_ALIGNMENT_END_MAD

        MAP_QUAL_DIFF_MEDIAN_FILTER = abs(np.median(read_mapping_qualities_in_tumor) - np.median(read_mapping_qualities_in_normal)) > c.MAX_MAP_QUAL_DIFF_MEDIAN
        VARIANT_MAP_QUAL_MEDIAN_FILTER = np.median(read_mapping_qualities_in_tumor) < c.MIN_VARIANT_MAP_QUAL_MEDIAN
        VARIANT_BASE_QUAL_MEDIAN_FILTER = np.median(base_qualities_in_tumor) < c.MIN_VARIANT_BASE_QUAL_MEDIAN

        if STRAND_BIAS_FILTER or ALLELE__FREQ_FILTER or MEDIAN_DISTANCE_TO_END_FILTER or MAD_DISTANCE_TO_END_FILTER or LOW_MAP_QUAL_READS_FILTER or MAP_QUAL_DIFF_MEDIAN_FILTER or VARIANT_MAP_QUAL_MEDIAN_FILTER or VARIANT_BASE_QUAL_MEDIAN_FILTER:
            continue
        else:
            positions.append(pos)
    
    # create folder for batches for this sample
    if not os.path.exists(os.path.join(c.filtering_folder, c.filtering_batches_folder, sample_name)):
        os.makedirs(os.path.join(c.filtering_folder, c.filtering_batches_folder, sample_name))

    if len(positions):
        # save batch.csv
        with open(batch_filename, 'a') as f:
            for pos in positions:
                f.write('%s\t%s\n' % (chrom, pos))

    print(("Saved batch %s" % batch_filename))

def ffpe_filter_snvs(bamname_n, bamname_t, snv_candidates_file):
    bamfile_n = pysam.AlignmentFile(bamname_n, 'rb')
    bamfile_t = pysam.AlignmentFile(bamname_t, 'rb')
  
    output_file = snv_candidates_file.replace('.csv','.npy')
    if os.path.exists(output_file):
        print('FFPE data file exists, deleting...', output_file)
        os.remove(output_file)
    
    data = {} 
    map_dict = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
 
    candidates = open(snv_candidates_file)
    candidates_list = []

    for line in candidates:
        line = line.strip().split()
        chrom, pos = line[0], int(line[1]) # 0-indexed snv pos
        candidates_list.append([chrom,pos])

    from random import shuffle
    shuffle(candidates_list)
        
    #candidates_list = candidates_list[:500000] # limit 500k
    
    for idx, site in enumerate(candidates_list):
        chrom, pos = site[0], site[1]
        coverage_n = bamfile_n.count_coverage(chrom, pos, pos+1, quality_threshold=c.MIN_BASE_QUALITY, read_callback = check_read)
        coverage_t = bamfile_t.count_coverage(chrom, pos, pos+1, quality_threshold=c.MIN_BASE_QUALITY, read_callback = check_read)

        # [ (#A, #C, #G, #T), (#A, #C, #G, #T), (#A, #C, #G, #T), ] at each position in normal
        coverage_list_n = [(coverage_n[0][i], coverage_n[1][i], coverage_n[2][i], coverage_n[3][i]) 
            for i in range(len(coverage_n[0]))]

        # [ (#A, #C, #G, #T), (#A, #C, #G, #T), (#A, #C, #G, #T), ] at each position in tumor
        coverage_list_t = [(coverage_t[0][i], coverage_t[1][i], coverage_t[2][i], coverage_t[3][i]) 
            for i in range(len(coverage_t[0]))]

        #print(chrom, pos)
        reads = get_reads(bamfile_t, chrom, pos, pos+1)

        #print('normal', coverage_list_n)
        #print('tumor', coverage_list_t)

        max_frequency_base_in_normal = max(enumerate( coverage_list_n[0] ), key=operator.itemgetter(1))
        normal_allele = map_dict[max_frequency_base_in_normal[0]]

        alt_allele, alt_allele_count = None, 0
        for j in range(len(coverage_list_t[0])):
                if j != max_frequency_base_in_normal[0]: # not the normal allele
                    if coverage_list_t[0][j] > alt_allele_count: # find alt allele with max count
                        alt_allele_count = coverage_list_t[0][j]
                        alt_allele = map_dict[j]

        #print('normal allele:', normal_allele)
        #print('alt allele:', alt_allele)
        
        alt_reads = []
        for read in reads:
            aligned_pairs = read.get_aligned_pairs()
            for p in aligned_pairs:
                read_pos, ref_pos = p[0], p[1]
                # check that p[0] is not None i.e. deletion
                if ref_pos == pos and read_pos is not None and read.query_sequence[read_pos] == alt_allele:
                    alt_reads.append(read)

        read1_alt_reads, forward_alt_reads = 0, 0
        for read in alt_reads:
            if read.is_read1:
                read1_alt_reads += 1
            if not read.is_reverse:
                forward_alt_reads += 1

        pos_key = 'chrom%spos%d' % (chrom, pos)
        data[pos_key] = {'normal_allele': normal_allele, 'alt_allele': alt_allele, 'alt_reads_count': len(alt_reads), 'read1_alt_reads': read1_alt_reads, 'forward_alt_reads': forward_alt_reads, 'total_tumor_reads': len(reads), 'coverage_list_n': coverage_list_n[0],\
        'coverage_list_t': coverage_list_t[0]}
        #print(data)
        
        if idx % 1000 == 0:
            # save every 1000 sites
            np.save(output_file, data)
            print('Saved:', output_file)
 
    np.save(output_file, data)   
    print('Saved:', output_file) 

def get_snv(coverage_list, REFERENCE_ALLELE):
    """
    coverage_list = [#A, #C, #G, #T] for site
    """
    ALLELES = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
    ALLELE_INDICES = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

    try:
        REFERENCE_ALLELE_COUNT = coverage_list[ALLELE_INDICES[REFERENCE_ALLELE]]
    except KeyError:
        # if the reference allele is ambiguous, 'N'
        REFERENCE_ALLELE_COUNT = 0

    DEPTH = sum(coverage_list)

    if REFERENCE_ALLELE in ALLELE_INDICES: # can be 'N'
        # find max count allele in tumor that is not reference allele
        alt_allele_count, alt_allele_index = -1, None
        for idx, i in enumerate(coverage_list):
            if i > alt_allele_count and idx != ALLELE_INDICES[REFERENCE_ALLELE]:
                alt_allele_count, alt_allele_index = i, idx

        ALT_ALLELE = ALLELES[alt_allele_index]
        ALT_ALLELE_READ_COUNT = alt_allele_count

        if DEPTH > 0:
            ALT_ALLELE_FRACTION = round(float(ALT_ALLELE_READ_COUNT)/float(DEPTH), 4)
        else:
            ALT_ALLELE_FRACTION = 0
    else:
        # ref allele is 'N' or something not ACGT, can't figure out ALT allele
        ALT_ALLELE = 'N'
        ALT_ALLELE_READ_COUNT, ALT_ALLELE_FRACTION = 0,0

    return (REFERENCE_ALLELE, ALT_ALLELE, DEPTH, REFERENCE_ALLELE_COUNT, ALT_ALLELE_READ_COUNT, ALT_ALLELE_FRACTION)

def filter_snvs(candidates_folder, bamname_n, bamname_t, ref_file, regions, batch_num, output_filename=None):

    output_file = os.path.join(candidates_folder, 'batch_%s.csv' % str(batch_num) )

    if os.path.exists(output_file):
        print(("SNV BATCH COMPELTE:", output_file))
        return
    
    if bamname_n:
        bamfile_n = pysam.AlignmentFile(bamname_n, 'rb')
    else:
        # tumor-only mode
        bamfile_n = None

    bamfile_t = pysam.AlignmentFile(bamname_t, 'rb')

    candidates = []

    for region in regions:
        chrom, start, end = region[0], region[1], region[2]

        reference_bases = get_ref_base(start, chrom, ref_file, end_pos=end + 1)

        try:
            if bamfile_n:
                coverage_n = bamfile_n.count_coverage(chrom, start, end+1, quality_threshold=c.MIN_BASE_QUALITY, read_callback = check_read)
            else:
                # tumor-only mode
                coverage_n = None

        except ValueError:
            if chrom == 'MT':
                # MT is chrM in hg19
                chrom = 'chrM'
            else:
                chrom = 'chr%s' % chrom

            try:
                coverage_n = bamfile_n.count_coverage(chrom, start, end+1, quality_threshold=c.MIN_BASE_QUALITY, read_callback = check_read)
            except ValueError:
                print("Region does not exist in normal BAM")
                return

        try:
            coverage_t = bamfile_t.count_coverage(chrom, start, end+1, quality_threshold=c.MIN_BASE_QUALITY, read_callback = check_read)
        except ValueError:
            if chrom == 'MT':
                # MT is chrM in hg19
                chrom = 'chrM'
            else:
                chrom = 'chr%s' % chrom

            try:        
                coverage_t = bamfile_t.count_coverage(chrom, start, end+1, quality_threshold=c.MIN_BASE_QUALITY, read_callback = check_read)
            except ValueError:
                print("Region does not exist in tumor BAM")
                return

        if coverage_n:
            # [ (#A, #C, #G, #T), (#A, #C, #G, #T), (#A, #C, #G, #T), ] at each position in normal
            coverage_list_n = [(coverage_n[0][i], coverage_n[1][i], coverage_n[2][i], coverage_n[3][i]) 
                for i in range(len(coverage_n[0]))]
            del coverage_n
        else:
            # tumor-only mode
            coverage_list_n = None

        # [ (#A, #C, #G, #T), (#A, #C, #G, #T), (#A, #C, #G, #T), ] at each position in tumor
        coverage_list_t = [(coverage_t[0][i], coverage_t[1][i], coverage_t[2][i], coverage_t[3][i]) 
            for i in range(len(coverage_t[0]))]
        
        del coverage_t

        gc.collect()

        map_dict = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
        reverse_map_dict = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        
        for i in range(len(coverage_list_t)):
            pos_key = 'chrom%spos%d' % (chrom, start+i)

            # Skip if either normal or tumor coverage is low
            if coverage_list_n:
                coverage_of_normal = sum(coverage_list_n[i])
                if coverage_of_normal < c.MIN_COVERAGE:
                    continue

            coverage_of_tumor = sum(coverage_list_t[i])
            if coverage_of_tumor < c.MIN_COVERAGE:
                continue

            # see if a variant allele is found with more than threshold frequency in tumor
            # assuming homozygosity
            if coverage_list_n:
                max_frequency_base_in_normal = max(enumerate( coverage_list_n[i] ), key=operator.itemgetter(1)) # (max_index, max_element)
                baseline_allele = max_frequency_base_in_normal[0]
            else:
                # tumor-only mode, compare REFERENCE base instead
                if reference_bases[i] in reverse_map_dict:
                    baseline_allele = reverse_map_dict[reference_bases[i]]
                else:
                    # reference is 'N' or something other than A,T,G,C
                    baseline_allele = None

            REFERENCE_ALLELE = reference_bases[i]
            is_candidate_position = False

            """
               Filter positions where no alternate allele has freq > c.MIN_MUTANT_ALLELE_FREQUENCY_IN_TUMOR
               Filter positions where no alterate allele has more than 2 supporting reads
               Filter positions where no alternate allele is found in the normal sample with frequency less than c.MAX_ALTERNATIVE_ALLELE_FREQUENCY_IN_NORMAL    
            """

            for j in range(len(coverage_list_t[i])):

                if j == baseline_allele:
                    continue

                # checking alternate alleles in tumor

                allele_frequency_high = (coverage_list_t[i][j] / coverage_of_tumor) >= c.MIN_MUTANT_ALLELE_FREQUENCY_IN_TUMOR        
                allele_read_count_high = coverage_list_t[i][j] >= c.MIN_MUTANT_ALLELE_READS_IN_TUMOR
                is_potential_mutation = allele_frequency_high and allele_read_count_high

                if coverage_list_n:
                    # normal-tumor mode; check that normal AF for variant is low
                    allele_frequency_low_in_normal = (coverage_list_n[i][j] / coverage_of_normal) <= c.MAX_ALTERNATIVE_ALLELE_FREQUENCY_IN_NORMAL    
                    is_potential_mutation = is_potential_mutation and allele_frequency_low_in_normal

                if is_potential_mutation:
                    is_candidate_position = True
                    break

            if is_candidate_position:
                # get INFO for candidate variant site
                REFERENCE_ALLELE, ALT_ALLELE, DEPTH, REFERENCE_ALLELE_COUNT, ALT_ALLELE_READ_COUNT, ALT_ALLELE_FRACTION = get_snv(coverage_list_t[i], REFERENCE_ALLELE)
                candidates.append((chrom, start + i, REFERENCE_ALLELE, ALT_ALLELE, str(DEPTH), str(REFERENCE_ALLELE_COUNT), str(ALT_ALLELE_READ_COUNT), str(ALT_ALLELE_FRACTION)))

    # save batch.csv
    with open(output_file, 'w') as f:
        for pos in candidates:
            f.write('%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' % (pos[0], pos[1], pos[2], pos[3], pos[4], pos[5], pos[6], pos[7]))

    print(('COMPLETED SNV BATCH: ', output_file)) 
