from collections import defaultdict
import math
import pdb
import os
import pickle
from itertools import product
import struct
from collections import defaultdict
from functools import partial

class Hifhif(object):
    """
    A class used for finding patterns in raw bytes that correspond to human-readable text.
    Given a number of seed phrases, it will search through all the files and look for possible mappings of the raw
    bytes in the file to the seed phrases.es.
    """

    CACHE_FILE = 'map.tbl'
    SEED_FILE = '__seed.txt'
    METADATA_FILE = '__metadata.txt'
    DATA_FOLDER = 'data'

    SPECIAL_CHARS = {
        'NEWLINE': '\n',
        'DOUBLENEWLINE': '\n\n',
    }

    CONTROL_CODE = 255

    def __init__(self, path, load=True):
        """
        The Hifhif class's only identifier is the location of all the files you wish to process.
        :param path: Absolute file path with the files to process
        :param load: If True, it will look for and attempt to load previously complete dictionaries.
        """

        self.path = path
        self.table = dict()
        self.files = [file_name for file_name in os.listdir(path) if file_name not in [self.CACHE_FILE, self.SEED_FILE,
                                                                                       self.DATA_FOLDER, self.METADATA_FILE]]
        if load:
            self.load()

    def save(self):
        table_file_lines = []
        for byte_code in sorted(self.table.keys()):
            table_file_lines.append(f'{byte_code}={self.table[byte_code]}')

        with open(os.path.join(self.path, self.DATA_FOLDER, self.CACHE_FILE), 'w', encoding='utf-8') as fh:
            fh.write('\n'.join(table_file_lines))

    def load(self):
        try:
            with open(os.path.join(self.path, self.DATA_FOLDER, self.CACHE_FILE), encoding='utf-8') as fh:
                table_file_lines = fh.readlines()
        except FileNotFoundError:
            print('No cached results were located for this directory')
            return

        loaded_table = {}
        for line in table_file_lines:
            byte_code, char = line.strip('\n').split('=')[:2]
            char = self.SPECIAL_CHARS.get(char, char)
            byte = int(byte_code)

            loaded_table[byte] = char

        self.table = loaded_table

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
        table = self.table.copy()
        control_object = Chikuhouse(text, table)

        new_text_chars = []
        counter = 0
        while counter < len(text):
            counter_increment = 1
            byte = text[counter]
            if byte == Chikuhouse.CONTROL_CODE:

                text_to_append, counter_increment = control_object.process_control_code(counter)
                new_text_chars.append(text_to_append)
            else:
                new_text_chars.append(table.get(byte, f'[{byte}]' if replace_unknown else byte))

            counter += counter_increment

        new_text = ''.join(new_text_chars)
        if output:
            with open(os.path.join(self.path, self.DATA_FOLDER, file_name + '.txt'), 'w', encoding='utf-8') as fh:
                fh.write(new_text)

        return new_text

    def get_text(self, file_name):
        with open(os.path.join(self.path, file_name), 'rb') as fh:
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

class Chikuhouse(object):
    """
    This object takes care of handling control code logic for Nazo Nazo Q
    """

    CONTROL_CODE = 255

    KNOWN_CONTROL_CODES = {
        (0,): ['continue_text', ()],
        (17, 1): ['new_character_text', ()],
        (35,): ['capture_until_next_control', ('*', '*')],
        (49,): ['capture_until_next_control', ('**', '**')],
        (10, 0): ['capture_arguments', (2,)]
    }


    def __init__(self, text, table):
        """
        Chikuhouse requires the body of text you'd like to search through as well as the table mapping, since some
        control sequences return text that is translatable.
        :param text:
        """
        assert isinstance(text, bytes)
        self.text = text
        self.table = table
        self.counter = 0

    def process_control_code(self, counter):
        """
        Given some text and the position of the control code indicated, return the replacement text and the number
        of steps to advance the text counter.
        """

        assert self.text[counter] == self.CONTROL_CODE
        self.counter = counter

        max_len = max([len(k) for k in self.KNOWN_CONTROL_CODES.keys()])

        for increment in range(1, max_len+1):
            start = counter + 1
            end = start + increment
            control_sequence = tuple(self.text[start:end])
            try:
                func_name, args = self.KNOWN_CONTROL_CODES[control_sequence]
                return self.__getattribute__('_' + func_name)(*args)
            except KeyError:
                continue
        else:
            # Unknown control sequence, just return the original control code
            return '[255]', 1

    def _continue_text(self):
        return '\n\n', 2

    def _new_character_text(self):
        return '\n----\n', 3

    def _capture_until_next_control(self, left_delimiter, right_delimiter):
        captured_text = []

        start_point = self.counter + 2
        for increment in range(1000):
            byte = self.text[start_point + increment]
            if byte == self.CONTROL_CODE:
                text_to_return = left_delimiter + ''.join(captured_text) + right_delimiter
                num_processed_chars = 3 + increment

                return text_to_return, num_processed_chars
            else:
                captured_text.append(self.table.get(byte, f'[{byte}]'))

        else:
            raise ValueError('Searched for an ending control point but couldn\'t find one!')

    def _capture_arguments(self, num_args):

        start_point = self.counter + 3
        end_point = start_point + num_args

        captured_args = ', '.join(str(x) for x in self.text[start_point:end_point])
        text_to_return = f'\n------NEW BOX ({captured_args})------\n'

        return text_to_return, 3 + num_args














    def _load_metadata(self):

        metadata_file = os.path.join(self.loader.path, self.loader.DATA_FOLDER, self.loader.METADATA_FILE)
        try:
            with open(metadata_file, 'r', encoding='utf-8') as fh:
               lines = fh.readlines()
        except FileNotFoundError:
            return {}

class BytesDecoder(object):
    """
    The main underlying class which handles replacement of bytes
    """


    def __init__(self, raw_bytes, table):
        self.bytes = raw_bytes
        self.table = table

    def replace_bytes(self, replacements):
        """
        Replacement should be a list of (needle, replacement) tuples
        :param replacements:
        :return:
        """
        if replacements is None:
            return

        raw_bytes = self.bytes

        text, correspondence_dict = self.bytes_to_text(self.bytes)
        for needle, replacement in replacements:

            position = self.find_phrase(text, needle)
            if position is None:
                raise ValueError(f'Wasn\'t able to locate phrase "{needle}"')

            corresponding_bytes = self.find_corresponding_bytes(correspondence_dict, *position)
            start = self.bytes.find(corresponding_bytes)
            assert start > -1
            end = start + len(corresponding_bytes)

            new_bytes = self.translate_phrase(replacement)

            raw_bytes = raw_bytes[0:start] + new_bytes + raw_bytes[end:]

        return raw_bytes

    def bytes_to_text(self, replace_unknown=True, as_bytes=False):
        new_text_chars = []
        counter = 0
        text_len_counter = 0
        byte_correspondence_dict = {}
        control_object = Chikuhouse(self.bytes, self.table)

        while counter < len(self.bytes):
            counter_increment = 1
            byte = self.bytes[counter]

            if as_bytes:
                text_to_append = f'[{byte}]'
            elif byte == Chikuhouse.CONTROL_CODE:
                text_to_append, counter_increment = control_object.process_control_code(counter)
            else:
                text_to_append = self.table.get(byte, f'[{byte}]' if replace_unknown else byte)

            new_text_chars.append(text_to_append)

            byte_correspondence_dict[text_len_counter] = (
                self.bytes[counter:counter + counter_increment], text_to_append)

            counter += counter_increment
            text_len_counter += len(text_to_append)

        new_text = ''.join(new_text_chars)
        return new_text, byte_correspondence_dict

    def find_phrase(self, text, phrase):
        start_index = text.find(phrase)
        if start_index < 0:
            return None

        return (start_index, start_index + len(phrase))

    def translate_phrase(self, phrase):
        reverse_map = {v:k for k, v in self.table.items()}

        byte_phrase = []
        for char in phrase:
            byte_phrase.append(reverse_map[char])

        return bytes(byte_phrase)

    def find_corresponding_bytes(self, correspondence_dict, start_index, end_index):

        def rollback(n):
            while n >= 0:
                try:
                    correspondence_dict[n]
                    return n
                except KeyError:
                    n -= 1
            else:
                raise IndexError('No compatible index found')

        all_indices = range(rollback(start_index), rollback(end_index))
        my_bytes = []
        for index in all_indices:
            try:
                for byte in correspondence_dict[index][0]:
                    my_bytes.append(byte)
            except KeyError:
                continue

        return bytes(my_bytes)

class PointerTableDecoder(object):

    def __init__(self, raw_bytes, table, offset=0, metadata=None, include_endpoints = False, **kwargs):
        self.bytes = raw_bytes
        self.table = table
        self.offset = offset
        self.pointer_map = {}
        self.include_endpoints = include_endpoints

        self.DELIMITER = f'---- POINTER TABLE ({offset}) ----'

        if metadata is None:
            metadata = {}

        pointers = parse_pointer_table(raw_bytes[8 if include_endpoints else 0:])
        for start, end in zip(pointers, pointers[1:] + [None]):
            if include_endpoints:
                start += 8
                if end is not None:
                    end += 8

            decoder_class = metadata.get(start + offset, BytesDecoder)
            kwargs = {}
            if decoder_class is PointerTableDecoder:
                kwargs = dict(offset=start, include_endpoints=True)

            self.pointer_map[start] = decoder_class(self.bytes[start:end], table, **kwargs)

    def replace_bytes(self, replacement_dict):
        if replacement_dict is None:
            return self.bytes

        pointers = []
        all_pointers = sorted(self.pointer_map.keys())

        byte_output = bytes()

        base_offset = len(all_pointers) * 4     # Implicit hardcode

        current_offset = 0
        for pointer in all_pointers:

            absolute_pointer = self.offset + pointer
            pointers.append(base_offset + current_offset)

            decoder_object = self.pointer_map[pointer]
            if isinstance(decoder_object, BytesDecoder):
                replacements = replacement_dict.get(absolute_pointer, [])
            else:
                replacements = replacement_dict

            bytes_to_append = decoder_object.replace_bytes(replacements)

            byte_output += bytes_to_append
            current_offset += len(bytes_to_append)

        new_bytes = reconstruct_pointer_table(pointers) + byte_output
        new_bytes += bytes([0]) * ((4 - len(new_bytes) % 4) % 4)

        if self.include_endpoints:
            new_bytes = struct.pack('<I', 8) + struct.pack('<I', len(new_bytes) + 8) + new_bytes



        return new_bytes

    def bytes_to_text(self, replace_unknown=True, as_bytes=False):
        text = self.DELIMITER + '\n'
        all_pointers = sorted(self.pointer_map.keys())

        for pointer in all_pointers:

            absolute_pointer = self.offset + pointer
            text += f'Pointer: {absolute_pointer}\n'
            text += self.pointer_map[pointer].bytes_to_text(replace_unknown=replace_unknown, as_bytes=as_bytes)[0]
            text += '\n\n'
        return text, None


class FileDecoder(object):

    DECODER_MAP = {
        'PointerTable': PointerTableDecoder,
        'Bytes': BytesDecoder
    }

    def __init__(self, path):
        self.loader = Hifhif(path, load=True)
        self.table = self.loader.table
        self.decoders = {}
        self.current_file = None
        self.metadata = defaultdict(dict)
        self._load_metadata()

    def open(self, file_name):
        raw_bytes = self.loader.get_text(file_name)

        self.decoders[file_name] = PointerTableDecoder(raw_bytes, self.table, metadata=self.metadata[file_name])
        self.current_file = file_name

    def bytes_to_text(self, replace_unknown=True, as_bytes=False):
        text = self.decoders[self.current_file].bytes_to_text(replace_unknown=replace_unknown, as_bytes=as_bytes)[0]
        suffix = '{}.txt'.format('.bytes' if as_bytes else '')
        output_file_name = os.path.join(self.loader.path, self.loader.DATA_FOLDER, self.current_file + suffix)
        with open(output_file_name, 'w', encoding='utf-8') as fh:
            fh.write(text)

        print(f'Output to: {output_file_name}')

    def replace_and_save(self, file_name = None):
        file_name = file_name or self.current_file
        new_file_name = os.path.join(self.loader.path, self.loader.DATA_FOLDER, 'RECONSTRUCTED_' + self.current_file)
        with open(new_file_name, 'wb') as fh:
            fh.write(self.decoders[file_name].replace_bytes(self._load_replacement_data()))
        print(f'Output to: {new_file_name}')


    def _load_metadata(self):
        metadata_file = os.path.join(self.loader.path, self.loader.DATA_FOLDER, self.loader.METADATA_FILE)
        with open(metadata_file, 'r', encoding='utf-8') as fh:
            lines = fh.readlines()

        current_file = None
        for line in lines:
            line = line.strip()
            if line.startswith('#'):
                continue
            if line.startswith('!'):
                current_file = line[1:]
                continue
            if ':' in line:
                pointer, decoder_class = line.split(':')
                pointer = int(pointer)
                self.metadata[current_file][pointer] = self.DECODER_MAP[decoder_class]

    def _load_replacement_data(self, file_name=None):
        if file_name is None:
            file_name = self.current_file

        rpl_file_name = os.path.join(self.loader.path, self.loader.DATA_FOLDER, file_name + '.rpl')
        try:
            with open(rpl_file_name, 'r', encoding='utf-8') as fh:
                lines = fh.readlines()
        except FileNotFoundError:
            print('No replacement data found for file: ' + rpl_file_name)
            return {}

        substitutions = defaultdict(list)
        current_pointer = None
        for line in lines:
            line = line.strip()
            if line.startswith('#'):
                continue
            if line.startswith('!'):
                current_pointer = int(line[1:])
                continue
            if '|' in line:
                needle, replacement = line.split('|')
                substitutions[current_pointer].append((needle, replacement))

        return substitutions


def parse_pointer_table(raw_bytes):

    format_code = '<I'
    int_size = 4

    pointers = []

    current_loc = 0
    while not len(pointers) or current_loc < pointers[0]:
        new_pointer = struct.unpack(format_code, raw_bytes[current_loc:current_loc + int_size])[0]
        pointers.append(new_pointer)
        current_loc += int_size

    return pointers

def reconstruct_pointer_table(pointers, format_code='<I', offset=0):

    raw_bytes = bytes()
    for pointer in pointers:
        raw_bytes += struct.pack(format_code, pointer + offset)

    return raw_bytes