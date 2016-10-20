# OP_RETURN.py
#
# Python script to generate and retrieve OP_RETURN peercoin transactions
#
# Copyright (c) Coin Sciences Ltd
# Copyright (c) 2016 Peerchemist <peerchemist@protonmail.ch>
# modified by hrobeers to work with ppcoin v0.5.4
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import argparse
import base64, random, binascii, struct, string, re, hashlib

# Python 2-3 compatibility logic
try:
    basestring
except NameError:
    basestring = str

OP_RETURN_BTC_FEE=0.01 # BTC fee to pay per transaction
OP_RETURN_BTC_DUST=0.00001 # omit BTC outputs smaller than this
OP_RETURN_MAX_BYTES=80 # maximum bytes in an OP_RETURN (40 as of Bitcoin 0.10) this lib can handle up to 65536 bytes
OP_RETURN_STORE_SPLIT=False # Splitting of data if longer than OP_RETURN_MAX_BYTES
OP_RETURN_MAX_BLOCKS=10 # maximum number of blocks to try when retrieving data


# User-facing functions
def send(node, send_address, send_amount, metadata):
    '''validats send_address and metadata, assembles the transaction and executes it.'''
    
    # Validate some parameters
    if not metadata:
        raise ValueError("metadata you want to write is null.")

    if isinstance(metadata, basestring): ## check for py/py2.7 compatibility
        metadata = metadata.encode('utf-8') # convert to binary string
    
    if len(metadata) > 65536:
        return {'error': 'This library only supports metadata up to 65536 bytes in size'}
    
    if len(metadata) > OP_RETURN_MAX_BYTES:
        return {'error': 'Metadata has ' + str(len(metadata)) + ' bytes but is limited to ' + str(OP_RETURN_MAX_BYTES) + ' (see OP_RETURN_MAX_BYTES)'}
    
    # Calculate amounts and choose inputs
    output_amount = send_amount + OP_RETURN_BTC_FEE
    
    # find apropriate inputs
    inputs, total_sum, change_address = TX_utils.select_inputs(output_amount)
    # change
    change_amount = total_sum - output_amount
    
    ## Build the raw transaction
    outputs={send_address: send_amount}
    
    if change_amount>=OP_RETURN_BTC_DUST:
        outputs[change_address] = change_amount

    raw_txn = TX_utils.create_txn(inputs, outputs, metadata)
    # Sign and send the transaction, return result
    return TX_utils.sign_send_txn(raw_txn)

def store(node, data):
    # Data is stored in OP_RETURNs within a series of chained transactions.
    # If the OP_RETURN is followed by another output, the data continues in the transaction spending that output.
    # When the OP_RETURN is the last output, this also signifies the end of the data.
    
    if isinstance(data, basestring):
        data=data.encode('utf-8') # convert to binary string
    
    if not data:
        return {'error': 'Some data is required to be stored'}
    
    # Calculate amounts and choose first inputs to use
    output_amount = OP_RETURN_BTC_FEE * int((data_len+OP_RETURN_MAX_BYTES-1) / OP_RETURN_MAX_BYTES) # number of transactions required
    
    # find apropriate inputs
    inputs, total_sum, change_address = TX_utils.select_inputs(output_amount)
    
    # Find the current blockchain height and mempool txids
    height = int(node.getblockcount())
    avoid_txids = node.getrawmempool()
    
    # Check if output splitting is supported
    if not OP_RETURN_STORE_SPLIT and len(data) > OP_RETURN_MAX_BYTES:
        return {'error': 'Data too large & splitting disabled, set OP_RETURN_STORE_SPLIT=True to split the data over multiple transactions.'}
    
    # Loop to build and send transactions
    result={'txids':[]}
    
    for data_ptr in range(0, data_len, OP_RETURN_MAX_BYTES):
        
        # Some preparation for this iteration
        last_txn=((data_ptr+OP_RETURN_MAX_BYTES)>=data_len) # is this the last tx in the chain?
        change_amount=input_amount-OP_RETURN_BTC_FEE
        metadata=data[data_ptr:data_ptr+OP_RETURN_MAX_BYTES]

        # Build and send this transaction
        outputs={}
        if change_amount>=OP_RETURN_BTC_DUST: # might be skipped for last transaction
            outputs[change_address]=change_amount

        raw_txn = TX_utils.create_txn(inputs, outputs, metadata, len(outputs) if last_txn else 0)

        send_result = TX_utils.sign_send_txn(raw_txn)

        # Check for errors and collect the txid
        if 'error' in send_result:
            result['error']=send_result['error']
            break

        result['txids'].append(send_result['txid'])

        if data_ptr == 0:
            result['ref'] = Data_utils.calc_ref(height, send_result['txid'], avoid_txids)

        # Prepare inputs for next iteration
        inputs=[{
            'txid': send_result['txid'],
            'vout': 1,
            }]

        input_amount=change_amount
    
    return result # Return the final result

def retrieve(node, ref, max_results=1):

    max_height = int(node.getblockcount())
    heights = Data_utils.get_ref_heights(ref, max_height)

    if not isinstance(heights, list):
        return {'error': 'Ref is not valid'}

    # Collect and return the results
    results = []

    for height in heights:
        if height == 0:
            txids = TX_utils.list_mempool_txns() # if mempool, only get list for now (to save RPC calls)
            txns = None
        else:
            txns = TX_utils.get_block_txns(height, testnet) # if block, get all fully unpacked
            txids = txns.keys()

    for txid in txids:
        if Data_utils.match_ref_txid(ref, txid):
            if height == 0:
                txn_unpacked = TX_utils.get_mempool_txn(txid)
        else:
            txn_unpacked = txns[txid]

        found = Data_utils.find_txn_data(txn_unpacked)

        if found:
            # Collect data from txid which matches ref and contains an OP_RETURN

            result = {
                'txids': [str(txid)],
                'data': found['op_return'],
                }

            key_heights = {height: True}

            if height == 0: # Work out which other block heights / mempool we should try
                try_heights = [] # nowhere else to look if first still in mempool

            else:
                result['ref'] = Data_utils.calc_ref(height, txid, txns.keys())
                try_heights = Data_utils.get_try_heights(height + 1, max_height, False)

            if height == 0: # Collect the rest of the data, if appropriate
                this_txns = TX_utils.get_mempool_txns() # now retrieve all to follow chain
            else:
                this_txns = txns

            last_txid = txid
            this_height = height

            while found['index'] < (len(txn_unpacked['vout'])-1): # this means more data to come
                next_txid = TX_utils.find_spent_txid(this_txns, last_txid, found['index'] + 1)

                # If we found the next txid in the data chain
                if next_txid:
                    result['txids'].append(str(next_txid))

                    txn_unpacked = this_txns[next_txid]
                    found = Data_utils.find_txn_data(txn_unpacked)

                    if found:
                        result['data'] += found['op_return']
                        key_heights[this_height] = True
                    else:
                        result['error'] = 'Data incomplete - missing OP_RETURN'
                        break

                    last_txid = next_txid

                else: # Otherwise move on to the next height to keep looking
                    if len(try_heights):
                        this_height = try_heights.pop(0)

                        if this_height == 0:
                            this_txns = TX_utils.get_mempool_txns()
                        else:
                            this_txns = TX_utils.get_block_txns(this_height, testnet)

                    else:
                        result['error'] = 'Data incomplete - could not find next transaction'
                        break

            # Finish up the information about this result
            result['heights'] = list(key_heights.keys())
            results.append(result)

            if len(results) >= max_results:
                break # stop if we have collected enough

    return results


# Utility functions
class TX_utils:

    @classmethod
    def validate_address(cls, address):
        assert node.validateaddress(address)["isvalid"] == True

    @classmethod
    def select_inputs(cls, total_amount):
        '''finds apropriate utxo's to include in rawtx, while being careful
        to never spend old transactions with a lot of coin age'''
        '''Argument is intiger, returns list of apropriate transactions'''
        from operator import itemgetter

        vins = []
        utxo_sum = float(-0.01) ## starts from negative due to fee
        for i in sorted(node.listunspent(), key=itemgetter('confirmations')):
            vins.append({"txid":i["txid"].encode(),"vout":i["vout"]})
            utxo_sum = utxo_sum + float(i["amount"])
            if utxo_sum >= total_amount:
                 #return txids
                change_address = i["address"].encode()
                return vins, utxo_sum, change_address
        if utxo_sum < total_amount:
            raise ValueError("Not enough funds.")

    @classmethod
    def create_txn(cls, inputs, outputs, metadata):

        raw_txn = node.createrawtransaction(inputs, outputs)
        txn_unpacked = cls.unpack_txn(Data_utils.hex_to_bin(raw_txn))

        if len(metadata) <= 252:  # 1 byte used for variable int , format uint_8
            data = b'\x4c' + struct.pack("B",len(metadata)) + metadata # OP_PUSHDATA1 format
        elif len(metadata) <= 65536:
            data = b'\x4d' + struct.pack('<H',len(metadata)) + metadata # OP_PUSHDATA2 format
        elif len(metadata) <= 4294967295:
            data = b'\x4e' + struct.pack('<L',len(metadata)) + metadata # OP_PUSHDATA4 format
        else:
            return {'error': 'metadata exceeds maximum length.'}
        
        txn_unpacked["vout"].append(
            {"value": 0, "scriptPubKey": "6a" + Data_utils.bin_to_hex(data)})

        return Data_utils.bin_to_hex(cls.pack_txn(txn_unpacked))
    
    @classmethod
    def sign_send_txn(cls, raw_txn):

        from math import ceil as cl
        
        signed_txn = node.signrawtransaction(raw_txn)
        if not ('complete' in signed_txn and signed_txn['complete']):
            return {'error': 'Could not sign the transaction'}

        # Check if the peercoin transaction fee is sufficient to cover the txn (0.01PPC/kb)
        txn_size = len(signed_txn['hex']) / 2 # 2 hex chars per byte
        if (txn_size / 1000 > OP_RETURN_BTC_FEE * 100):
            return {'error': 'Transaction fee too low to be accepted on the peercoin chain. Required fee: ' + str(cl(txn_size / 1024) * 0.01) + ' PPC'}

        send_txid = node.sendrawtransaction(signed_txn["hex"])
        if not (isinstance(send_txid, basestring) and len(send_txid) == 64):
            return {'error': 'Could not send the transaction'}

        return {'txid': str(send_txid)}
    
    @classmethod
    def list_mempool_txns(cls):
        return node.getrawmempool()
    
    @classmethod
    def get_mempool_txn(cls, txid):
        raw_txn = node.getrawtransaction(txid)
        return cls.unpack_txn(Data_utils.hex_to_bin(raw_txn))
    
    @classmethod
    def get_mempool_txns(cls):
        txids = cls.list_mempool_txns()

        txns={}
        for txid in txids:
            txns[txid] = cls.get_mempool_txn(txid)

        return txns
    
    @classmethod
    def get_raw_block(cls, block_height):

        block_hash = node.getblockhash(block_height)
        if not (isinstance(block_hash, basestring) and len(block_hash) == 64):
            return {'error': 'Block at height ' + str(height) + ' not found'}

        return {
            'block': Data_utils.hex_to_bin(node.getblock(block_hash))
        }

    @classmethod
    def get_block_txns(cls, block_height):
        raw_block = cls.get_raw_block(block_height)
        if 'error' in raw_block:
            return {'error': raw_block['error']}

        block=cls.unpack_block(raw_block['block'])

        return block['txs']
     
    @classmethod
    def unpack_txn(cls, binary):
        return cls.unpack_txn_buffer(OP_RETURN_buffer(binary))
    
    @classmethod
    def pack_txn(cls, txn):
        binary=b''

        binary += struct.pack('<L', txn['version'])
        # peercoin: 4 byte timestamp https://wiki.peercointalk.org/index.php?title=Transactions
        binary += struct.pack('<L', txn['timestamp'])

        binary += cls.pack_varint(len(txn['vin']))

        for input in txn['vin']:
            binary += Data_utils.hex_to_bin(input['txid'])[::-1]
            binary += struct.pack('<L', input['vout'])
            binary += cls.pack_varint(int(len(input['scriptSig']) / 2 )) # divide by 2 because it is currently in hex
            binary += Data_utils.hex_to_bin(input['scriptSig'])
            binary += struct.pack('<L', input['sequence'])

        binary += cls.pack_varint(len(txn['vout']))

        for output in txn['vout']:
            binary += Data_utils.pack_uint64(int(round(output['value'] * 1000000 )))
            binary += Data_utils.pack_varint(int(len(output['scriptPubKey']) / 2 )) # divide by 2 because it is currently in hex
            binary += Data_utils.hex_to_bin(output['scriptPubKey'])

        binary += struct.pack('<L', txn['locktime'])

        return binary

    @classmethod
    def unpack_block(cls, binary):
        buffer=OP_RETURN_buffer(binary)
        block={}

        block['version'] = buffer.shift_unpack(4, '<L')
        block['hashPrevBlock'] = Data_utils.bin_to_hex(buffer.shift(32)[::-1])
        block['hashMerkleRoot'] = Data_utils.bin_to_hex(buffer.shift(32)[::-1])
        block['time'] = buffer.shift_unpack(4, '<L')
        block['bits'] = buffer.shift_unpack(4, '<L')
        block['nonce'] = buffer.shift_unpack(4, '<L')
        block['tx_count'] = buffer.shift_varint()

        block['txs'] = {}

        old_ptr = buffer.used()

        while buffer.remaining():
            transaction = cls.unpack_txn_buffer(buffer)
            new_ptr = buffer.used()
            size = new_ptr-old_ptr

            raw_txn_binary = binary[old_ptr:old_ptr + size]
            txid = Data_utils.bin_to_hex(hashlib.sha256(hashlib.sha256(raw_txn_binary).digest()).digest()[::-1])

            old_ptr = new_ptr
            transaction['size'] = size
            block['txs'][txid] = transaction

        return block
    
    @classmethod
    def unpack_txn_buffer(cls, buffer):
        # see: https://en.bitcoin.it/wiki/Transactions

        txn={
            'vin': [],
            'vout': [],
        }

        txn['version'] = buffer.shift_unpack(4, '<L') # small-endian 32-bits
        # peercoin: 4 byte timestamp https://wiki.peercointalk.org/index.php?title=Transactions
        txn['timestamp'] = buffer.shift_unpack(4, '<L') # small-endian 32-bits

        inputs = buffer.shift_varint()
        if inputs > 100000: # sanity check
            return None

        for _ in range(inputs):
            _input = {}

            _input['txid'] = Data_utils.bin_to_hex(buffer.shift(32)[::-1])
            _input['vout'] = buffer.shift_unpack(4, '<L')
            length=buffer.shift_varint()
            _input['scriptSig'] = Data_utils.bin_to_hex(buffer.shift(length))
            _input['sequence'] = buffer.shift_unpack(4, '<L')

            txn['vin'].append(_input)

        outputs = buffer.shift_varint()
        if outputs > 100000: # sanity check
            return None

        for _ in range(outputs):
            output={}

            output['value']=float(buffer.shift_uint64()) / 1000000
            length=buffer.shift_varint()
            output['scriptPubKey'] = Data_utils.bin_to_hex(buffer.shift(length))

            txn['vout'].append(output)

        txn['locktime'] = buffer.shift_unpack(4, '<L')

        return txn
    
    @classmethod
    def find_spent_txid(cls, txns, spent_txid, spent_vout):

        for txid, txn_unpacked in txns.items():
            for input in txn_unpacked['vin']:
                if (input['txid'] == spent_txid) and (input['vout'] == spent_vout):
                    return txid
        return None

# Working with data references

# The format of a data reference is: [estimated block height]-[partial txid] - where:

# [estimated block height] is the block where the first transaction might appear and following
# which all subsequent transactions are expected to appear. In the event of a weird blockchain
# reorg, it is possible the first transaction might appear in a slightly earlier block. When
# embedding data, we set [estimated block height] to 1+(the current block height).

# [partial txid] contains 2 adjacent bytes from the txid, at a specific position in the txid:
# 2*([partial txid] div 65536) gives the offset of the 2 adjacent bytes, between 0 and 28.
# ([partial txid] mod 256) is the byte of the txid at that offset.
# (([partial txid] mod 65536) div 256) is the byte of the txid at that offset plus one.
# Note that the txid is ordered according to user presentation, not raw data in the block.

class Data_utils:
    
    @classmethod
    def calc_ref(cls, next_height, txid, avoid_txids):
        txid_binary = cls.hex_to_bin(txid)

        for txid_offset in range(15):
            sub_txid=txid_binary[2*txid_offset:2*txid_offset+2]
            clashed=False

            for avoid_txid in avoid_txids:
                avoid_txid_binary = cls.hex_to_bin(avoid_txid)

            if (
                (avoid_txid_binary[2*txid_offset:2*txid_offset+2]==sub_txid) and
                (txid_binary!=avoid_txid_binary)
            ):
                clashed=True
                break

            if not clashed:
                break

        if clashed: # could not find a good reference
            return None

        tx_ref=ord(txid_binary[2*txid_offset:1+2*txid_offset])+256*ord(txid_binary[1+2*txid_offset:2+2*txid_offset])+65536*txid_offset

        return '%06d-%06d' % (next_height, tx_ref)
    
    @classmethod
    def get_ref_parts(cls, ref):
        if not re.search('^[0-9]+\-[0-9A-Fa-f]+$', ref): # also support partial txid for second half
            return None

        parts=ref.split('-')

        if re.search('[A-Fa-f]', parts[1]):
            if len(parts[1]) >= 4:
                txid_binary = cls.hex_to_bin(parts[1][0:4])
                parts[1]=ord(txid_binary[0:1])+256*ord(txid_binary[1:2])+65536*0
            else:
                return None

        parts=list(map(int, parts))

        if parts[1]>983039: # 14*65536+65535
            return None

        return parts
    
    @classmethod
    def get_ref_heights(cls, ref, max_height):
        parts = cls.get_ref_parts(ref)
        if not parts:
            return None

        return cls.get_try_heights(parts[0], max_height, True)
    
    @classmethod
    def get_try_heights(cls, est_height, max_height, also_back):
        forward_height = est_height
        back_height = min(forward_height-1, max_height)

        heights=[]
        mempool=False
        try_height=0

        while True:
            if also_back and ((try_height%3)==2): # step back every 3 tries
                heights.append(back_height)
                back_height-=1
        
            else:
                if forward_height>max_height:
                    if not mempool:
                        heights.append(0) # indicates to try mempool
                        mempool=True

                    elif not also_back:
                        break # nothing more to do here

                else:
                    heights.append(forward_height)

            forward_height+=1

            if len(heights) >= OP_RETURN_MAX_BLOCKS:
                break

            try_height+=1

        return heights

    @classmethod
    def match_ref_txid(cls, ref, txid):
        parts = cls.get_ref_parts(ref)

        if not parts:
            return None

        txid_offset = int(parts[1] / 65536)
        txid_binary = cls.hex_to_bin(txid)

        txid_part=txid_binary[2 * txid_offset:2 * txid_offset + 2]
        txid_match=bytearray([parts[1]%256, int((parts[1]%65536)/256)])

        return txid_part == txid_match # exact binary comparison
    
    @classmethod
    def find_txn_data(cls, txn_unpacked):
        for index, output in enumerate(txn_unpacked['vout']):
            op_return = cls.get_script_data(cls.hex_to_bin(output['scriptPubKey']))

            if op_return:
                return {
                    'index': index,
                    'op_return': op_return,
                }

        return None

    @classmethod
    def get_script_data(cls, scriptPubKeyBinary):
        op_return=None

        if scriptPubKeyBinary[0:1] == b'\x6a':
            first_ord=ord(scriptPubKeyBinary[1:2])

            if first_ord <= 75:
                op_return = scriptPubKeyBinary[2:2+first_ord]
            elif first_ord == 0x4c:
                op_return = scriptPubKeyBinary[3:3+ord(scriptPubKeyBinary[2:3])]
            elif first_ord == 0x4d:
                op_return = scriptPubKeyBinary[4:4+ord(scriptPubKeyBinary[2:3])+256*ord(scriptPubKeyBinary[3:4])]

        return op_return
    
    @classmethod
    def pack_varint(cls, integer):
        if integer > 0xFFFFFFFF:
            packed = b'\xFF' + cls.pack_uint64(integer)
        elif integer > 0xFFFF:
            packed = b'\xFE'+struct.pack('<L', integer)
        elif integer > 0xFC:
            packed = b'\xFD'+struct.pack('<H', integer)
        else:
            packed = struct.pack('B', integer)

        return packed
    
    @classmethod
    def pack_uint64(cls, integer):
        upper = int(integer / 4294967296)
        lower = integer - upper * 4294967296

        return struct.pack('<L', lower) + struct.pack('<L', upper)

    # Converting binary <-> hexadecimal
    @classmethod
    def hex_to_bin(cls, hex):
        try:
            raw=binascii.a2b_hex(hex)
        except Exception:
            return None

        return raw
    
    @classmethod
    def bin_to_hex(cls, string):
        return binascii.b2a_hex(string).decode('utf-8')


class OP_RETURN_buffer(): # Helper class for unpacking bitcoin binary data

    def __init__(self, data, ptr=0):
        self.data=data
        self.len=len(data)
        self.ptr=ptr

    def shift(self, chars):
        prefix = self.data[self.ptr:self.ptr+chars]
        self.ptr += chars

        return prefix

    def shift_unpack(self, chars, format):
        unpack=struct.unpack(format, self.shift(chars))

        return unpack[0]

    def shift_varint(self):
        value=self.shift_unpack(1, 'B')

        if value == 0xFF:
            value = self.shift_uint64()
        elif value == 0xFE:
            value = self.shift_unpack(4, '<L')
        elif value == 0xFD:
            value = self.shift_unpack(2, '<H')

        return value

    def shift_uint64(self):
        return self.shift_unpack(4, '<L') + 4294967296 * self.shift_unpack(4, '<L')

    def used(self):
        return min(self.ptr, self.len)

    def remaining(self):
        return max(self.len-self.ptr, 0)


### ### ### ### ### ### ### ### ### ### ### ###

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Send and recive OP_RETURNs in Peercoin transactions.')
    #parser.add_argument("-version", help="print version.", action="store_true")
    parser.add_argument("-auth", help="Specify username/password for Peercoin JSON-RPC interface: <username>, <password>",
                        action="store_true")
    parser.add_argument("-testnet", help="Operate on Peercoin testnet.", action="store_true")
    parser.add_argument("-send", help="Send OP_RETURN message: <send-address> <send-amount> <message>", 
                        nargs="*")
    parser.add_argument("-store", help="Store some data on the blockchain: <data>", nargs="*")
    parser.add_argument("-retrieve", help="Retrieve OP_RETURN data from the blockchain.", nargs="*")
    args = parser.parse_args()

    from peercoin_rpc import Client # import Client only if called as a script

    if args.testnet:
        node = Client(testnet=True)
    else:
        node = Client(testnet=False)

    if args.testnet and args.auth:
        node = Client(testnet=True, username=args.auth[0], password=args.auth[1])

    if args.auth:
        node = Client(testnet=False, username=args.auth[0], password=args.auth[1])

    if args.send:
        result = send(node, str(args.send[0]), float(args.send[1]), str(args.send[2]))
        if 'error' in result:
            print('Error: ' + result['error'])
        else:
            print("Success: ", result)
    
    if args.retrieve:

        results = retrieve(node, ref, 1)

        if 'error' in results:
            print('Error: '+results['error'])
    
        elif len(results):
            for result in results:
                print("Hex: ("+str(len(result['data']))+" bytes)\n" + Data_utils.bin_to_hex(result['data'])+"\n")
                print("ASCII:\n"+re.sub(b'[^\x20-\x7E]', b'?', result['data']).decode('utf-8')+"\n")
                print("TxIDs: (count "+str(len(result['txids']))+")\n"+"\n".join(result['txids'])+"\n")
                print("Blocks:"+("\n"+("\n".join(map(str, result['heights'])))+"\n").replace("\n0\n", "\n[mempool]\n"))
                
                if 'ref' in result:
                    print("Best ref:\n"+result['ref']+"\n")

                if 'error' in result:
                    print("Error:\n"+result['error']+"\n")

        else:
            print("No matching data was found")

    if args.store:
        
        if Data_utils.hex_to_bin(args.store[0]) is not None:
            data = data_from_hex
            result = store(node, data)

        result = store(node, args.store[0])
        if 'error' in result:
            print('Error: ' + result['error'])
        else:
            print("TxIDs:\n"+"\n".join(result['txids']) + "\n\nRef: " + result['ref'])

