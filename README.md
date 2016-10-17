# peercoin-OP_RETURN

## Simple Python library for using OP_RETURNs in peercoin transactions.

*Forked from python-OP_RETURN by Coin Sciences Ltd.*

* Copyright (c) Coin Sciences Ltd - http://coinsecrets.org/
* Copyright (c) peerchemist       - https://github.com/peerchemist
* Copyright (c) hrobeers          - https://github.com/hrobeers

MIT License (see headers in files)

_____________________________________________________________________

REQUIREMENTS
------------
* Python 2.5 or later (including Python 3)
* ppcoin 0.5.4 or later
* [peercoin_rpc](https://github.com/peerchemist/peercoin_rpc)


BEFORE YOU START
----------------
Check the constant settings at the top of OP_RETURN.py.
If you have just installed peercoin, wait for it to download and verify old blocks.
If using as a library, you need to import `peercoin_rpc` as well:

`from OP_RETURN import *`

`from peercoin_rpc import Client`

you need to create a new `Client` instance to enable communication with `ppcoind`.

`ppc_node = Client(testnet=True, username="username", password="password")`


TO SEND A PEERCOIN TRANSACTION WITH SOME OP_RETURN METADATA
-----------------------------------------------------------

On the command line:

* `python OP_RETURN.py -send <send-address> <send-amount> <metadata> (-testnet)`

  ```
  <send-address> is the peercoin address of the recipient
  <send-amount> is the amount to send (in units of PPC)
  <metadata> is a hex string or raw string containing the OP_RETURN metadata
             (auto-detection: treated as a hex string if it is a valid one)
  -testnet use testnet
  ```

* Outputs an error if one occurred or the txid if sending was successful

* Examples:

  ```
  python OP_RETURN.py -send PCrtba6CAGBB7ztsVGhrAnCs6hvt4t6Pep 0.001 'Hello, blockchain!'
  python OP_RETURN.py -send PCrtba6CAGBB7ztsVGhrAnCs6hvt4t6Pep 0.001 48656c6c6f2c20626c6f636b636861696e21
  python OP_RETURN.py -send -testnet mzEJxCrdva57shpv62udriBBgMECmaPce4 0.001 'Hello, testnet!'
  ```

As a library:

* `send(ppc_node, send_address, send_amount, metadata)`

  ```
  send_address is the peercoin address of the recipient
  send_amount is the amount to send (in units of BTC)
  metadata is a string of raw bytes containing the OP_RETURN metadata
  ```

* Returns: {'error': '<some error string>'}
       or: {'txid': '<sent txid>'}

* Examples

  ```
  send(ppc_node, 'PCrtba6CAGBB7ztsVGhrAnCs6hvt4t6Pep', 0.001, 'Hello, blockchain!')
  send(ppc_node, 'mzEJxCrdva57shpv62udriBBgMECmaPce4', 0.001, 'Hello, testnet!')
  ```


TO STORE SOME DATA IN THE BLOCKCHAIN USING OP_RETURNs
-----------------------------------------------------

On the command line:

* `python OP_RETURN.py -store <data>`

  ```
  <data> is a hex string or raw string containing the data to be stored
         (auto-detection: treated as a hex string if it is a valid one)
  -testnet use the peercoin testnet
  ```

* Outputs an error if one occurred or if successful, the txids that were used to store
  the data and a short reference that can be used to retrieve it using this library.

* Examples:

  ```
  python OP_RETURN.py -store 'This example stores 47 bytes in the blockchain.'
  python OP_RETURN.py -store -testnet 'This example stores 44 bytes in the testnet.'
  ```

As a library:

* `store(ppc_node, data)`
  
  ```
  <data> is the string of raw bytes to be stored
  -testnet is whether to use the peercoin testnet network (False if omitted)
  ```

* Returns: {'error': '<some error string>'}
       or: {'txids': ['<1st txid>', '<2nd txid>', ...],
            'ref': '<ref for retrieving data>'}

* Examples:

  ```
  store(ppc_node, 'This example stores 47 bytes in the blockchain.')
  ```

TO RETRIEVE SOME DATA FROM OP_RETURNs IN THE BLOCKCHAIN
-------------------------------------------------------

On the command line:

* `python OP_RETURN.py -retrieve <ref>`

  ```
  <ref> is the reference that was returned by a previous storage operation
  -testnet use the peercoin testnet
  ```

* Outputs an error if one occurred or if successful, the retrieved data in hexadecimal
  and ASCII format, a list of the txids used to store the data, a list of the blocks in
  which the data is stored, and (if available) the best ref for retrieving the data
  quickly in future. This may or may not be different from the ref you provided.

* Examples:

  ```
  python OP_RETURN.py -retrieve 356115-052075
  python OP_RETURN.py -testnet -retrieve 396381-059737
  ```

As a library:

* `retrieve(ppc_node, ref, max_results=1)`

  ```
  <ref> is the reference that was returned by a previous storage operation
  <max_results> is the maximum number of results to retrieve (in general, omit for 1)
  -testnet is whether to use the peercoin testnet network (False if omitted)
  ```

* Returns: {'error': '<some error string>'}
       or: {'data': '<raw binary data>',
            'txids': ['<1st txid>', '<2nd txid>', ...],
            'heights': [<block 1 used>, <block 2 used>, ...],
            'ref': '<best ref for retrieving data>',
            'error': '<error if data only partially retrieved>'}

           A value of 0 in the 'heights' array means some data is still in the mempool.
           The 'ref' and 'error' elements are only present if appropriate.

* Examples:

  ```
  retrieve(ppc_node, '356115-052075')
  retrieve(ppc_node, '396381-059737', 1)
  ```


VERSION HISTORY
---------------
v2.0.2 - 30 June 2015
* First port of php-OP_RETURN to Python

peercoin-OP_RETURN - 28 July 2016 (hrobeers)
* Code ported to work on the Peercoin blockchain

peercoin-OP_RETURN 0.21 - 15 Oct 2016 (peerchemist)
*  Significant code clenup and redesign.
