from collections import defaultdict
import math
import pdb
import os
import pickle
from itertools import product

# TODO:
# Figure out how to handle new lines/new text boxes
# Write some code that, given a verified dictionary and a new dictionary, checks for consistency and merges if there's
# a singular match
# Save down master dictionaries somewhere

class Hifhif(object):
    """
    A class used for finding patterns in raw bytes that correspond to human-readable text.
    Given a number of seed phrases, it will search through all the files and look for possible mappings of the raw
    bytes in the file to the seed phrases.es.
    """

    CACHE_FILE = 'map.tbl'
    SEED_FILE = '__seed.txt'
    DATA_FOLDER = 'data'

    def __init__(self, path, load=True):
        """
        The Hifhif class's only identifier is the location of all the files you wish to process.
        :param path: Absolute file path with the files to process
        :param load: If True, it will look for and attempt to load previously complete dictionaries.
        """

        self.path = path
        self.table = dict()
        self.files = [file_name for file_name in os.listdir(path) if file_name not in [self.CACHE_FILE, self.SEED_FILE]]
        if load:
            self.load()

    def save(self):
        table_file_lines = []




        print ('Change this to output a table file')
        with open(os.path.join(self.path, self.DATA_FOLDER, self.CACHE_FILE), 'wb') as fh:
            pickle.dump(self.table, fh)

    def load(self):
        print ('Change this to read a table file')
        try:
            with open(os.path.join(self.path, self.DATA_FOLDER, self.CACHE_FILE), 'rb') as fh:
                self.table = pickle.load(fh)
        except FileNotFoundError:
            print('No cached results were located for this directory')

    def update_seeds(self, seeds, overwrite = False):
        """
        Updates the seed list
        :param overwrite: If set to True, will erase the existing seed file
        :return: Nothing.
        """
        if isinstance(seeds, str):
            seeds = [seeds]

        if not overwrite:
            current_seeds = self.retrieve_seeds()
        else:
            current_seeds = []

        for seed in seeds:
            if seed not in current_seeds:
                current_seeds.append(seed)

        with open(os.path.join(self.path, self.DATA_FOLDER, self.SEED_FILE), 'w', encoding='utf-8') as fh:
            fh.write('\n'.join(current_seeds))

    def retrieve_seeds(self):
        try:
            with open(os.path.join(self.path, self.DATA_FOLDER, self.SEED_FILE), encoding='utf-8') as fh:
                raw_text = fh.read()
                seeds = [seed.strip().strip('\ufeff').replace('\u3000', ' ')
                         for seed in raw_text.split('\n')]
                seeds = [seed for seed in seeds if seed]
            return seeds

        except FileNotFoundError:
            print (f'Could not find a {self.SEED_FILE} file')
            return []

    def translate_text(self, file_name, replace_unknown = True, output = True):
        """
        With the existing dictionary mapping, translate all text that is currently available.
        :param file_name:
        :param chunk_size:
        :return:
        """
        text = self.get_text(file_name)
        reverse_mapping = {v: k for k, v in self.table.items()}
        # Hack! Keeping in newlines since I know what those are
        reverse_mapping['\n'] = '\n'

        new_text_chars = []
        for char in text:
            new_text_chars.append(reverse_mapping.get(char, f'[{ord(char)}]' if replace_unknown else char))

        new_text = ''.join(new_text_chars)
        if output:
            with open(os.path.join(self.path, self.DATA_FOLDER, file_name + '.txt'), 'w', encoding='utf-8') as fh:
                fh.write(new_text)

        return new_text

    def get_text(self, file_name):
        with open(os.path.join(self.path, file_name), 'r', errors='ignore') as fh:
            text = fh.read()
        return text


    def search_for_seeds(self, file_name, chunk_size = 1):
        """
        The main function of this class. Given the seeds in the seed file, will look through a file for instances of
        this pattern. If it finds a single consistent mapping for all, it will update the master table.

        :param file_name: The file you wish to search through.
        :param chunk_size:
        :return: A summary of mappings for all files.
        """

        seeds = self.retrieve_seeds()
        if not len(seeds):
            raise ValueError('You haven\'t loaded any seeds to search for! Please add some using update_seeds.')

        text = self.get_text(file_name)
        existing_mapping = self.table.copy()

        all_potential_proposals = [existing_mapping]
        for seed in seeds:
            new_potential_proposals = []

            for potential_proposal in all_potential_proposals:

                proposals = self.locate_potential_patterns(seed, text, chunk_size=chunk_size,
                                                           validation_dict=potential_proposal)
                proposals = self.remove_duplicate_dicts(proposals)

                new_potential_proposals.extend(proposals)

            all_potential_proposals = new_potential_proposals

        if len(all_potential_proposals) == 1:
            print(f'Found consistent mapping for {file_name}!')
            self.table = all_potential_proposals[0]
        elif len(all_potential_proposals) > 1:
            print(f'Found multiple consistent mappings for {file_name}! Try adding more seeds.')
            pdb.set_trace()

    def merge_proposals(self, proposals_by_seed):
        """
        This function takes a list of potential mappings and attempts to create consistent merged mappings.
        :param proposals: A list of potential mappings corresponding to each seed.
        :return: A list of potential merged mappings.
        """

        merged_results = []

        for mapping_set in product(*proposals_by_seed):
            initial_mapping = {}
            for mapping in mapping_set:
                if self.dicts_are_consistent(initial_mapping, mapping, enforce_equality=False):
                    initial_mapping.update(mapping)
                else:
                    break
            else:
                merged_results.append(initial_mapping)
        return merged_results


    @staticmethod
    def dicts_are_consistent(dict1, dict2, enforce_equality=False):

        if enforce_equality:
            if set(dict1.keys()) != set(dict2.keys()):
                return False

        reverse_dict1 = {v: k for k, v in dict1.items()}
        reverse_dict2 = {v: k for k, v in dict2.items()}

        for d1, d2 in [[dict1, dict2], [reverse_dict1, reverse_dict2]]:
            intersecting_keys = set(d1.keys()).intersection(set(d2.keys()))
            for key in intersecting_keys:
                if d1[key] != d2[key]:
                    return False

        return True

    @staticmethod
    def remove_duplicate_dicts(dict_list):
        # Remove duplicate proposals
        final_list = []
        for proposal in dict_list:
            for valid_proposal in final_list:
                if Hifhif.dicts_are_consistent(proposal, valid_proposal, enforce_equality=True):
                    break
            else:
                final_list.append(proposal)
        return final_list


    @staticmethod
    def chunk_text(text, chunk_size):
        return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

    @staticmethod
    def locate_potential_patterns(pattern, text, chunk_size=1, validation_dict = None):
        """
        This function does the heavy lifting of identifying potential byte->pattern character mappings. Given a pattern
        and a body of text to search through, it will return a list of potential mappings that match.
        :param pattern: A string indicating the pattern you're looking for
        :param text: A string for the body of text you're searching for
        :param chunk_size: The size of the blocks which should be matched.
        :param validation_dict: A pre-existing dictionary which the proposed patterns must conform to. If a mapping
            disagrees with the input dictionary, the result is discarded.
        :return:
        """

        pattern_data = defaultdict(list)
        chunked_pattern = []
        paren_mode = False
        current_paren_chunk = ''
        for char in pattern:
            if not paren_mode:
                if char == '(':
                    paren_mode = True
                else:
                    chunked_pattern.append(char)
            else:
                if char == ')':
                    paren_mode = False
                    chunked_pattern.append(current_paren_chunk)
                    current_paren_chunk = ''
                else:
                    current_paren_chunk += char

        if paren_mode:
            raise ValueError('You have an unfinished paren in your pattern!')

        for i in range(len(chunked_pattern)):
            # We take note of the characters, but we also take note of special instances
            key = chunked_pattern[i]
            pattern_data[key].append(i)

        chunks = Hifhif.chunk_text(text, chunk_size)
        length = len(chunked_pattern)

        if validation_dict is None:
            validation_dict = {}

        existing_chars = set(validation_dict.values())
        possible_mappings = []

        for iteration in range(len(chunks)):
            start_pt = iteration
            end_pt = start_pt + length

            if end_pt > len(chunks):  # Lazy, I know...
                break

            chunks_to_search = chunks[start_pt:end_pt]
            
            claimed_chunks = set()
            proposed_mapping = dict()

            for char, char_locations in pattern_data.items():

                proposed_chunk = None
                for location in char_locations:

                    potential_chunk = chunks_to_search[location]  # Agreement means we check the rest of the locations
                    # If the character doesn't already have a mappi
                    if potential_chunk == proposed_chunk:
                        continue
                    elif proposed_chunk is None:  # For the initial loop
                        proposed_chunk = potential_chunk
                    else:  # The proposed chunks do not agree, this part of the text cannot be a match to our pattern
                        proposed_chunk = False
                        break
                if proposed_chunk is False:
                    break
                else:
                    # If another character has already mapped to this chunk, we fail and move on
                    if proposed_chunk in claimed_chunks:
                        break

                    # Chunk <-> character mappings must be one to one.
                    # If we detect that the chunk maps to a different character already, or if we detect that
                    # the character in question has a different chunk mapping to it, we discard this case.
                    validation_char = validation_dict.get(proposed_chunk)
                    if (validation_char is not None or char in existing_chars) and char != validation_char:
                        break

                    claimed_chunks.add(proposed_chunk)
                    proposed_mapping[proposed_chunk] = char
            else:
                # If all characters were successfully mapped to a chunk, we propose this mapping and move on.
                proposed_mapping.update(validation_dict)
                possible_mappings.append(proposed_mapping)

        return possible_mappings