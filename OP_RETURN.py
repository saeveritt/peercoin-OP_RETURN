# OP_RETURN.py
#
# Python script to generate and retrieve OP_RETURN peercoin transactions
#
# Copyright (c) Coin Sciences Ltd
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

from peercoin_rpc import Client

node = Client(testnet=True)

import subprocess, base64, json, time, random, os.path, binascii, struct, string, re, hashlib, math

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
OP_RETURN_NET_TIMEOUT=10 # how long to time out (in seconds) when communicating with peercoin node


# User-facing functions

def OP_RETURN_send(send_address, send_amount, metadata):
    
    # Validate some parameters
    assert node.validateaddress(send_address)["isvalid"] 
    
    if isinstance(metadata, basestring):
        metadata = metadata.encode('utf-8') # convert to binary string
    
    if len(metadata) > 65536:
        return {'error': 'This library only supports metadata up to 65536 bytes in size'}
    
    if len(metadata) > OP_RETURN_MAX_BYTES:
        return {'error': 'Metadata has ' + str(len(metadata)) + ' bytes but is limited to ' + str(OP_RETURN_MAX_BYTES) + ' (see OP_RETURN_MAX_BYTES)'}
    
    # Calculate amounts and choose inputs
    output_amount = send_amount + OP_RETURN_BTC_FEE
    
    # find apropriate inputs
    inputs = Utils.OP_RETURN_select_inputs(output_amount)
    # change
    change_amount = inputs['total'] - output_amount
    
    ## Build the raw transaction
    change_address = Utils.OP_RETURN_get_change_address(inputs['inputs'])
    
    outputs={send_address: send_amount}
    
    if change_amount>=OP_RETURN_BTC_DUST:
        outputs[change_address] = change_amount

    raw_txn = Utils.OP_RETURN_create_txn(inputs['inputs'], outputs, metadata, len(outputs))
    # Sign and send the transaction, return result
    return Utils.OP_RETURN_sign_send_txn(raw_txn)

def OP_RETURN_store(data, testnet=False):
    # Data is stored in OP_RETURNs within a series of chained transactions.
    # If the OP_RETURN is followed by another output, the data continues in the transaction spending that output.
    # When the OP_RETURN is the last output, this also signifies the end of the data.
    
    if isinstance(data, basestring):
        data=data.encode('utf-8') # convert to binary string
    
    if not data:
        return {'error': 'Some data is required to be stored'}
    
    # Calculate amounts and choose first inputs to use
    output_amount = OP_RETURN_BTC_FEE * int((data_len+OP_RETURN_MAX_BYTES-1) / OP_RETURN_MAX_BYTES) # number of transactions required
    
    inputs_spend = Utils.OP_RETURN_select_inputs(output_amount)
    if 'error' in inputs_spend:
        return {'error': inputs_spend['error']}
    
    inputs=inputs_spend['inputs']
    input_amount=inputs_spend['total']
    
    # Get the change_address
    change_address = Utils.OP_RETURN_get_change_address(inputs_spend['inputs'])
    
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

        raw_txn = Utils.OP_RETURN_create_txn(inputs, outputs, metadata, len(outputs) if last_txn else 0)

        send_result = Utils.OP_RETURN_sign_send_txn(raw_txn)

        # Check for errors and collect the txid
        if 'error' in send_result:
            result['error']=send_result['error']
            break

        result['txids'].append(send_result['txid'])

        if data_ptr==0:
            result['ref']=OP_RETURN_calc_ref(height, send_result['txid'], avoid_txids)

        # Prepare inputs for next iteration
        inputs=[{
            'txid': send_result['txid'],
            'vout': 1,
            }]

        input_amount=change_amount
    
    return result # Return the final result

def OP_RETURN_retrieve(ref, max_results=1, testnet=False):
    # Validate parameters and get status of Bitcoin Core
    
    max_height = int(node.getblockcount())
    heights=OP_RETURN_get_ref_heights(ref, max_height)
    
    if not isinstance(heights, list):
        return {'error': 'Ref is not valid'}
    
    # Collect and return the results
    results=[]

    for height in heights:
        if height == 0:
            txids = Utils.OP_RETURN_list_mempool_txns() # if mempool, only get list for now (to save RPC calls)
            txns=None
        else:
            txns=OP_RETURN_get_block_txns(height, testnet) # if block, get all fully unpacked
            txids=txns.keys()

    for txid in txids:
      if OP_RETURN_match_ref_txid(ref, txid):
        if height==0:
          txn_unpacked = Utils.OP_RETURN_get_mempool_txn(txid)
        else:
          txn_unpacked=txns[txid]

        found=OP_RETURN_find_txn_data(txn_unpacked)

        if found:
          # Collect data from txid which matches ref and contains an OP_RETURN

          result={
            'txids': [str(txid)],
            'data': found['op_return'],
          }

          key_heights={height: True}

          # Work out which other block heights / mempool we should try

          if height==0:
            try_heights=[] # nowhere else to look if first still in mempool
          else:
            result['ref']=OP_RETURN_calc_ref(height, txid, txns.keys())
            try_heights=OP_RETURN_get_try_heights(height+1, max_height, False)

          # Collect the rest of the data, if appropriate

          if height==0:
            this_txns = Utils.OP_RETURN_get_mempool_txns() # now retrieve all to follow chain
          else:
            this_txns=txns

          last_txid=txid
          this_height=height

          while found['index'] < (len(txn_unpacked['vout'])-1): # this means more data to come
            next_txid=OP_RETURN_find_spent_txid(this_txns, last_txid, found['index']+1)

            # If we found the next txid in the data chain

            if next_txid:
              result['txids'].append(str(next_txid))

              txn_unpacked=this_txns[next_txid]
              found=OP_RETURN_find_txn_data(txn_unpacked)

              if found:
                result['data']+=found['op_return']
                key_heights[this_height]=True
              else:
                result['error']='Data incomplete - missing OP_RETURN'
                break

              last_txid=next_txid

            # Otherwise move on to the next height to keep looking
            else:
              if len(try_heights):
                this_height=try_heights.pop(0)

                if this_height==0:
                  this_txns = Utils.OP_RETURN_get_mempool_txns()
                else:
                  this_txns=OP_RETURN_get_block_txns(this_height, testnet)

              else:
                result['error']='Data incomplete - could not find next transaction'
                break

          # Finish up the information about this result

          result['heights']=list(key_heights.keys())
          results.append(result)

        if len(results)>=max_results:
            break # stop if we have collected enough
    
    return results


# Utility functions

class Utils:

    @classmethod
    def OP_RETURN_select_inputs(cls, total_amount): ## drop-in replacement for former OP_RETURN_select_inputs
        '''finds apropriate utxo's to include in rawtx, while being careful
        to never spend old transactions with a lot of coin age'''
        '''Argument is intiger, returns list of apropriate transactions'''
        from operator import itemgetter
        
        txids = []
        utxo_sum = float(-0.01) ## starts from negative due to fee
        for i in sorted(node.listunspent(), key=itemgetter('confirmations')):
            if i["txid"] not in txids:
                txids.append(i["txid"])
                utxo_sum = utxo_sum + float(i["amount"])
                if utxo_sum >= total_amount:
                    #return txids
                    return {
                        'inputs': txids,
                        'total': utxo_sum,
                    }
                else:
                    raise ValueError("Not enough founds.")
    
    @classmethod
    def OP_RETURN_get_change_address(cls, inputs):
        try:
            return inputs[0]['address']
        except:
            return node.getnewaddress()
    
    @classmethod
    def OP_RETURN_create_txn(cls, inputs, outputs, metadata, metadata_pos):

        raw_txn = node.createrawtransaction(inputs, outputs)
        txn_unpacked = OP_RETURN_unpack_txn(OP_RETURN_hex_to_bin(raw_txn))

        if not metadata:
            raise ValueError
        elif len(metadata) <= 75:
            data = bytearray(len(metadata)) + metadata
        elif len(metadata) < 256:
            data = b'\x4c' + bytearray(len(metadata)) + metadata ## OP_PUSHDATA1 format
        elif len(metadata) < 65536:
            data = b'\x4c' + struct.pack('<H',len(metadata)) + metadata # OP_PUSHDATA2 format
        else:
            data = b'\x4c' + struct.pack('<L',len(metadata)) + metadata # OP_PUSHDATA4 format
        
        medadata_pos = min(max(0, metadata_pos)), len(txn_unpacked["vout"]) # constrain to valid values

        #txn_unpacked["vout"].append(
        #    {"value": 0, "scriptPubKey": "6a" + OP_RETURN_bin_to_hex(data)
        #    }) 

        txn_unpacked["vout"]["metadata_pos:medadata_pos"] = [{
            "value": 0,
            "scriptPubKey": "6a" + OP_RETURN_bin_to_hex(data) # here is the OP_RETURN
        }]

        return OP_RETURN_bin_to_hex(OP_RETURN_pack_txn(txn_unpacked))
    
    @classmethod
    def OP_RETURN_sign_send_txn(cls, raw_txn):

        signed_txn = node.signrawtransaction(raw_txn)
        if not ('complete' in signed_txn and signed_txn['complete']):
            return {'error': 'Could not sign the transaction'}

        # Check if the peercoin transaction fee is sufficient to cover the txn (0.01PPC/kb)
        txn_size = len(signed_txn['hex'])/2 # 2 hex chars per byte
        if (txn_size/1000 > OP_RETURN_BTC_FEE*100):
            return {'error': 'Transaction fee too low to be accepted on the peercoin chain. Required fee: ' + str(math.ceil(txn_size/1024) * 0.01) + ' PPC'}

        send_txid = node.sendrawtransaction(signed_txn["hex"])
        if not (isinstance(send_txid, basestring) and len(send_txid)==64):
            return {'error': 'Could not send the transaction'}

        return {'txid': str(send_txid)}
    
    @classmethod
    def OP_RETURN_list_mempool_txns(cls):
        return node.getrawmempool()
    
    @classmethod
    def OP_RETURN_get_mempool_txn(txid):
        raw_txn = node.getrawtransaction(txid)
        return OP_RETURN_unpack_txn(OP_RETURN_hex_to_bin(raw_txn))
    
    @classmethod
    def OP_RETURN_get_mempool_txns(cls):
        txids = cls.OP_RETURN_list_mempool_txns()

        txns={}
        for txid in txids:
            txns[txid] = cls.OP_RETURN_get_mempool_txn(txid)

        return txns
    
    @classmethod
    def OP_RETURN_get_raw_block(cls, block_height):

        block_hash = node.getblockhash(block_height)
        if not (isinstance(block_hash, basestring) and len(block_hash) == 64):
            return {'error': 'Block at height ' + str(height) + ' not found'}

        return {
            'block': OP_RETURN_hex_to_bin(node.getblock(block_hash))
    }

    @classmethod
    def OP_RETURN_get_block_txns(block_height):
        raw_block = cls.OP_RETURN_get_raw_block(block_height)
        if 'error' in raw_block:
            return {'error': raw_block['error']}

        block=OP_RETURN_unpack_block(raw_block['block'])

        return block['txs']

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


def OP_RETURN_calc_ref(next_height, txid, avoid_txids):
    txid_binary=OP_RETURN_hex_to_bin(txid)

    for txid_offset in range(15):
        sub_txid=txid_binary[2*txid_offset:2*txid_offset+2]
        clashed=False

        for avoid_txid in avoid_txids:
            avoid_txid_binary=OP_RETURN_hex_to_bin(avoid_txid)

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

def OP_RETURN_get_ref_parts(ref):
    if not re.search('^[0-9]+\-[0-9A-Fa-f]+$', ref): # also support partial txid for second half
        return None

    parts=ref.split('-')

    if re.search('[A-Fa-f]', parts[1]):
        if len(parts[1])>=4:
            txid_binary=OP_RETURN_hex_to_bin(parts[1][0:4])
            parts[1]=ord(txid_binary[0:1])+256*ord(txid_binary[1:2])+65536*0
        else:
            return None

    parts=list(map(int, parts))

    if parts[1]>983039: # 14*65536+65535
        return None

    return parts

def OP_RETURN_get_ref_heights(ref, max_height):
    parts=OP_RETURN_get_ref_parts(ref)
    if not parts:
        return None

    return OP_RETURN_get_try_heights(parts[0], max_height, True)

def OP_RETURN_get_try_heights(est_height, max_height, also_back):
    forward_height=est_height
    back_height=min(forward_height-1, max_height)

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

def OP_RETURN_match_ref_txid(ref, txid):
    parts=OP_RETURN_get_ref_parts(ref)

    if not parts:
        return None

    txid_offset=int(parts[1]/65536)
    txid_binary=OP_RETURN_hex_to_bin(txid)

    txid_part=txid_binary[2*txid_offset:2*txid_offset+2]
    txid_match=bytearray([parts[1]%256, int((parts[1]%65536)/256)])

    return txid_part==txid_match # exact binary comparison

## Unpacking and packing bitcoin blocks and transactions

def OP_RETURN_unpack_block(binary):
    buffer=OP_RETURN_buffer(binary)
    block={}

    block['version']=buffer.shift_unpack(4, '<L')
    block['hashPrevBlock']=OP_RETURN_bin_to_hex(buffer.shift(32)[::-1])
    block['hashMerkleRoot']=OP_RETURN_bin_to_hex(buffer.shift(32)[::-1])
    block['time']=buffer.shift_unpack(4, '<L')
    block['bits']=buffer.shift_unpack(4, '<L')
    block['nonce']=buffer.shift_unpack(4, '<L')
    block['tx_count']=buffer.shift_varint()

    block['txs']={}

    old_ptr=buffer.used()

    while buffer.remaining():
        transaction=OP_RETURN_unpack_txn_buffer(buffer)
        new_ptr=buffer.used()
        size=new_ptr-old_ptr

        raw_txn_binary=binary[old_ptr:old_ptr+size]
        txid=OP_RETURN_bin_to_hex(hashlib.sha256(hashlib.sha256(raw_txn_binary).digest()).digest()[::-1])

        old_ptr=new_ptr

        transaction['size']=size
        block['txs'][txid]=transaction

    return block

def OP_RETURN_unpack_txn(binary):
    return OP_RETURN_unpack_txn_buffer(OP_RETURN_buffer(binary))

def OP_RETURN_unpack_txn_buffer(buffer):
    # see: https://en.bitcoin.it/wiki/Transactions

    txn={
        'vin': [],
        'vout': [],
    }

    txn['version']=buffer.shift_unpack(4, '<L') # small-endian 32-bits
    # peercoin: 4 byte timestamp https://wiki.peercointalk.org/index.php?title=Transactions
    txn['timestamp']=buffer.shift_unpack(4, '<L') # small-endian 32-bits

    inputs=buffer.shift_varint()
    if inputs > 100000: # sanity check
        return None

    for _ in range(inputs):
        input={}

        input['txid']=OP_RETURN_bin_to_hex(buffer.shift(32)[::-1])
        input['vout']=buffer.shift_unpack(4, '<L')
        length=buffer.shift_varint()
        input['scriptSig']=OP_RETURN_bin_to_hex(buffer.shift(length))
        input['sequence']=buffer.shift_unpack(4, '<L')

        txn['vin'].append(input)

    outputs = buffer.shift_varint()
    if outputs > 100000: # sanity check
        return None

    for _ in range(outputs):
        output={}

        output['value']=float(buffer.shift_uint64())/1000000
        length=buffer.shift_varint()
        output['scriptPubKey']=OP_RETURN_bin_to_hex(buffer.shift(length))

        txn['vout'].append(output)

    txn['locktime']=buffer.shift_unpack(4, '<L')

    return txn

def OP_RETURN_find_spent_txid(txns, spent_txid, spent_vout):

    for txid, txn_unpacked in txns.items():
        for input in txn_unpacked['vin']:
            if (input['txid'] == spent_txid) and (input['vout'] == spent_vout):
                return txid
    return None

def OP_RETURN_find_txn_data(txn_unpacked):
    for index, output in enumerate(txn_unpacked['vout']):
        op_return=OP_RETURN_get_script_data(OP_RETURN_hex_to_bin(output['scriptPubKey']))

        if op_return:
            return {
                'index': index,
                'op_return': op_return,
            }

    return None

def OP_RETURN_get_script_data(scriptPubKeyBinary):
    op_return=None

    if scriptPubKeyBinary[0:1]==b'\x6a':
        first_ord=ord(scriptPubKeyBinary[1:2])

        if first_ord <= 75:
            op_return = scriptPubKeyBinary[2:2+first_ord]
        elif first_ord == 0x4c:
            op_return = scriptPubKeyBinary[3:3+ord(scriptPubKeyBinary[2:3])]
        elif first_ord == 0x4d:
            op_return = scriptPubKeyBinary[4:4+ord(scriptPubKeyBinary[2:3])+256*ord(scriptPubKeyBinary[3:4])]

    return op_return

def OP_RETURN_pack_txn(txn):
    binary=b''

    binary+=struct.pack('<L', txn['version'])
    # peercoin: 4 byte timestamp https://wiki.peercointalk.org/index.php?title=Transactions
    binary+=struct.pack('<L', txn['timestamp'])

    binary+=OP_RETURN_pack_varint(len(txn['vin']))

    for input in txn['vin']:
        binary+=OP_RETURN_hex_to_bin(input['txid'])[::-1]
        binary+=struct.pack('<L', input['vout'])
        binary+=OP_RETURN_pack_varint(int(len(input['scriptSig'])/2)) # divide by 2 because it is currently in hex
        binary+=OP_RETURN_hex_to_bin(input['scriptSig'])
        binary+=struct.pack('<L', input['sequence'])

    binary+=OP_RETURN_pack_varint(len(txn['vout']))

    for output in txn['vout']:
        binary+=OP_RETURN_pack_uint64(int(round(output['value']*1000000)))
        binary+=OP_RETURN_pack_varint(int(len(output['scriptPubKey'])/2)) # divide by 2 because it is currently in hex
        binary+=OP_RETURN_hex_to_bin(output['scriptPubKey'])

    binary+=struct.pack('<L', txn['locktime'])

    return binary

def OP_RETURN_pack_varint(integer):
    if integer>0xFFFFFFFF:
        packed=b'\xFF'+OP_RETURN_pack_uint64(integer)
    elif integer>0xFFFF:
        packed=b'\xFE'+struct.pack('<L', integer)
    elif integer>0xFC:
        packed=b'\xFD'+struct.pack('<H', integer)
    else:
        packed=struct.pack('B', integer)

    return packed

def OP_RETURN_pack_uint64(integer):
    upper=int(integer/4294967296)
    lower=integer-upper*4294967296

    return struct.pack('<L', lower)+struct.pack('<L', upper)

# Helper class for unpacking bitcoin binary data
class OP_RETURN_buffer():

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

# Converting binary <-> hexadecimal
def OP_RETURN_hex_to_bin(hex):
    try:
        raw=binascii.a2b_hex(hex)
    except Exception:
        return None

    return raw

def OP_RETURN_bin_to_hex(string):
    return binascii.b2a_hex(string).decode('utf-8')
