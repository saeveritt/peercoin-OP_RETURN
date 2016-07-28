# store-OP_RETURN.py
#
# CLI wrapper for OP_RETURN.py to store data using OP_RETURNs
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


import sys, string
from OP_RETURN import *


if len(sys.argv)<2:
  sys.exit(
'''Usage:
python store-OP_RETURN.py <data> <testnet (optional)>

<data> is a hex string or raw string containing the OP_RETURN metadata
       (auto-detection: treated as a hex string if it is a valid one)
<testnet> should be 1 to use the peercoin testnet, otherwise it can be omitted

To store the contents of a file, use:
python store-OP_RETURN.py "$(cat <filepath>)" <testnet (optional)>
'''
  )

data=sys.argv[1]

if len(sys.argv)>2:
  testnet=bool(sys.argv[2])
else:
  testnet=False

data_from_hex=OP_RETURN_hex_to_bin(data)
if data_from_hex is not None:
  data=data_from_hex

result=OP_RETURN_store(data, testnet)

if 'error' in result:
  print('Error: '+result['error'])
else:
  print("TxIDs:\n"+"\n".join(result['txids'])+"\n\nRef: "+result['ref']+"\n\nWait a few seconds then check on: http://"+
    ('testnet.' if testnet else '')+'coinsecrets.org/')
