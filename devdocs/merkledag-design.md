# DyND Support for CAS and Merkle DAGs

## Motivation

When dealing with data in a distributed context, it turns out to be extremely useful to refer to data by its content, represented as a hash of the data. This allows one to answer questions like "do machines X and Y have the same data," or when the data is further decomposed into a Merkle DAG, "how can we efficiently synchronize data updates from X to Y." It also allows for compact and verifiable labeling of computation results, e.g. "X is the result of operation Y on data Z".

The core technology here is called a _Content Addressable Storage_ (CAS), which is an associative array `{hash(X) : X}` using a cryptographic hash function. With every blob of data `X` now having a short, immutable, self-authenticating name `hash(X)`, we can build a data structure called a _Merkle DAG_. In a Merkle DAG, each node is a blob `X` within the CAS, and contains links to other nodes `Y_i` using their names `hash(Y_i)`.

The scope of DyND does not include full-blown implementations of these, but a system providing CAS and Merkle DAGs must be built out of something. Large numerical data can be represented as Merkle DAGs built out of smaller chunks, and DyND is all about representing and manipulating such chunks. Providing basic functionality to support the creation of systems for large numerical data is definitely within scope, and that is what is being proposed here.

## What is Required for a Merkle DAG?

See https://github.com/ipfs/specs/tree/master/merkledag for an example of specifying a Merkle DAG.

Why not simply adopt the IPFS Merkle DAG? Because it seems natural to define things in terms of [DyND serialization](serialization-design.md) instead of protocol buffers. For storage of raw data like files and DyND numerical data, storing just the data and not tagging it or packing it into some more complicated serialization format provides for the best opportunistic sharing between similar systems. In the design outlined below, a file is always stored untouched, so its hash is the hash of the file in the most straightforward way. Similarly the DyND serialization is designed so that it is an obvious "raw" data format in common cases, and that is stored directly.

### Content Addressable Storage

First, you need a CAS. The CAS specifies how the hash of a particular blob of data is computed. For example, within `git` the blobs stored in its CAS can have one of several types, and that type is prepended to the data. For example, file blobs begin with ascii "blob &lt;size&gt;\0", so to compute the git hash of a file "Hello, World!", you can execute (http://stackoverflow.com/questions/7225313/how-does-git-compute-file-hashes)

```
$ echo -e 'blob 14\0Hello, World!' | shasum
8ab686eafeb1f44702738c8b0f24f2567c36da6d
```

In picking how to encode data into bytes, and how to insert those bytes into a CAS, this form of prefixing the bytes with a type and size does not seem like a good idea. Having the base CAS be the simple associative array `{ hash(data) : data }` feels like a really good idea, extra interpretation of the data can be on the side or implemented in terms of this.

### Hash Links

Graphs (in our case Directed Acyclic Graphs or DAGs) are made up of nodes and edges. A node is simply an entry in the CAS, whose data is in a particular format. An edge from `node1` to `node2` is simply the value `hash(node2)` stored within `node1`. This hash may be encoded in a hexadecimal format if it is embedded within text data, or as binary data. For encoding hash links, the `multihash` protocol (https://github.com/jbenet/multihash) looks like a great idea.

The CAS in git is used to store multiple types of DAGs, such as directory trees and commit histories. This is an important thing to support, and users of a DyND-based Merkle DAG system will likely need to define new ones.

### Node Data Format

Being able to work with the Merkle DAG as a generic graph is a very powerful idea, and enables the sharing of many operations between different use cases. To enable this, the data format used for nodes must be standardized in a way that the Merkle DAG can be understood as a graph without any special knowledge of the particular data that is stored within. For example, synchronization of subgraphs between computers, determining which data was used to compute a particular result, finding all the ways a particular bit of data has been used, and many more things can be implemented generically.

For most node types, simply having a header section that stores the node structure in XML, JSON, or a similar format can suffice. For typed DyND array data which may also contain hash links, we will either have to come up with a serialization form that allows for following the links without understanding the data type, or require DyND be present in that case.

## Types of Merkle DAG

To help clarify some of the needs of hash links, let's run through some of the different DAGs we might store.

### Directory Structures

This is simple, consisting of two types of links, 'file' and 'directory'. Each 'file' link points at a leaf node which is interpreted as strictly binary data, while each 'directory' link points at a node containing zero or more 'file' and 'directory' links. Git includes this, and designing the specifics of directory structures for the DyND Merkle DAG should be made with reference to the choices there. File links would typically also incorporate file metadata like attributes, permissions, etc.

### Version History

Any application with an undo/redo system tracks a series of versions of each document being edited.  Git is a particular, sophisticated application of this idea, using the DAG to store a history of directory structures with branching and merging of versions. A node supporting version history will contain zero or more 'predecessor' links, which represent prior versions of whatever the node contains. Such a node would also typically contain metadata such as a timestamp, an author, etc.

### Document Graphs

These are also known as Scene Graphs or Document Object Models. In an application built to edit a document, exploding its graph into the CAS can provide a lot of benefits. Modifications are often localized to a small set of nodes, and the result of such a changed document graph will share a lot of data with the previous version. The directory structure described above can actually be thought of as a special case of a document graph. The link types will depend on the particular document graph being represented, but supporting a way to tag these links as part of a document graph would be useful for generic algorithms operating on the Merkle DAG.

### Expression DAGs

Expressions can be formulated as trees, and if there are common subexpressions, those can be merged together and result in a DAG. An expression tree formulated as a Merkle DAG automatically will have this property by construction, if no incidental information like a timestamp is added. A node in an expression DAG can be thought of as a function application on immutable arguments, like in functional programming. Each child node is either another function application, or contains raw data.

Each expression node can be evaluated into raw data, and that data can then be inserted into the CAS as well. Because of variations in computation libraries, such as different implementations of math primitives and reordering of almost-associative operations in parallel code, there will not be a unique mapping from an expression node to an evaluated result.

A problem which arises from this is that the query "what are all the evaluated versions of this expression node" cannot be answered efficiently with just the CAS. This is similar to queries like "what are all the successor versions of a node" in a version history, and a general mechanism to indicate when such information might be queried and acceleration structures for it will be necessary.

### Raw Numerical Data / DyND Arrays

When storing raw numerical data in the CAS, a unique serialization of that data is highly desirable, both to deduplicate that data and to make the query "where are all the places this data is used" possible. The [DyND serialization design document](serialization-design.md) specifies a serialization protocol with the intent to be useful in this context.

A possible extension to DyND is to make a Merkle link into a first class type that operates similar to the type `array[T]`. This could be as `merkle[T]`, and would interact with a CAS registered into the DyND runtime. For example, an array including such Merkle links could be lazily loaded. It could support an initial unhashed state, where it behaves exactly like `array[T]`, and a commit/checkpoint operation would hash the data and save it to the CAS.

## Potential Merkle DAG System Features

The strength of creating a Merkle DAG system lies in the ability to write generic algorithms on all the various forms of Merkle DAG in a generic way, and provide those as a service to applications. With an application's data expressed in standard formats, things like working with version history, synchronizing data in a distributed computation or with the cloud, and exploring the input data that went into producing a particular computed result can be done via calls into the generic service. Supporting this constrains the design of such a generic Merkle DAG system, so let's enumerate what features we may want, and how they affect the design.

### Extracting Merkle Sub-DAGS

* Extract all the nodes linked into a particular node (its full set of dependencies).
* Extract all the nodes of the current version of a document, ignoring all history links.
* Extract just the "source data", throwing away all links to raw numerical data that was derived from a computation.
* Extract just the "derived data", throwing away all links to how it was computed.

This seems to be effectively a traversal of all Merkle Links, running a predicate `should_follow(node, link)` on each link. The predicates need to be able to see attributes of the nodes and links. Some likely attributes for nodes are:

* Node type (directory structure node, document root, expression node, dynd array node, expression evaluation result node)
* Authorship metadata (name, email, creation timestamp, etc)

Some likely attributes for links are:

* The name of the link within the node
* The node type of what it points to (e.g. whether the link points to a generic Merkle DAG node, etc.)
* File attributes like mime-type, permissions for files stored as binary blobs
* Whether the link points at an interior node which itself contains Merkle links (e.g. with a DyND type containing a Merkle link, such links would exist in nodes that are not themselves generic Merkle DAG nodes)
* Whether the link points to a previous version of the same node

### Synchronizing a Set of Nodes Across a Network

Some examples of this include pushing a set of nodes to the cloud or a compute cluster to run a computation there, pulling a set of nodes from remote storage for viewing locally, or updating a backup server with newly created data. For git, `git push` and `git fetch` are doing a version of this. See [HyperOS](http://hyperos.io/), in particular its hyperfs, for a pretty sophisticated application of this idea. Also see [Max Ogden's Merkle Graph Synchronization Draft](https://gist.github.com/maxogden/9ebd17dc839f065d12f6).

Just like with the `git` transfer protocol, it makes sense to compute what needs to be sent from the source server, possibly by sending it a Merkle Sub-DAG predicate, and then use a smart protocol to send just the necessary nodes across the wire. There may be better and worse orderings to send things in, for example first sending low level of detail data, or data most visible in a specified viewport, so that the user sees a useful result as quickly as possible.

### Walking the Merkle DAG in Reverse

Given just the hash of a leaf data node, it may be useful to query how it's being used. For example, to find all the expression evaluation result nodes, to determine which expressions evaluated to the value. To find all the expressions which use the node as an input. Taking that a step further, find all the rendered charts that used that data is part of their inputs.

To support this, the CAS system needs to calculate and save all the reverse links.

### Derived Metadata of Nodes

As part of creating the reverse link cache, additional metadata about each node could be cached as well. For example, a DyND array link with type `T` and data `D` implies that `T` can be interpreted as a DyND type, and that the data `D` can be validly interpreted as a serialized DyND array of type `T`. A given node can have multiple such interpretations, for example the data `[0x00]` could be the `string` "", the `bool` false, the `int8` 0, among other things.

## Proposed DyND Merkle DAG Features

### Merkle Data Link

This adds a first class Merkle link type into DyND, as well as defining a CAS interface where these links may be loaded or saved. The data for a Merkle link consists of a `bytes` hash and an `nd::array` reference. The type is `merkle[T]`, where `T` has the same meaning as in `array[T]`, being a pattern the type of any data linked must conform to. The possible states of such a link are:

1. The hash is empty and the nd::array reference is NULL.
   * The initial state of a Merkle link, not referring to any data.
2. The hash is empty, but the nd::array reference contains data.
   * The data is in a mutable state, the array pointed at or the nd::array reference may be modified.
3. The hash is non-empty, but the nd::array reference is NULL.
   * This is for lazy loading. The nd::array reference is only allowed be set to data which matches the hash, for example by triggering a load from a CAS interface.
4. The hash is non-empty, and the nd::array contains data.
   * The data must be immutable, the result of saving/freezing from state 2 or loading from a CAS.

The main operations defined for working with Merkle links include:

* Load a specific array from a CAS, given the type and data hashes. Should be able to specify recursive (recursively load all Merkle links) or not (lazy loading).
* Freeze an array and everything it references. Freezing from mutable to immutable requires ensuring full ownership, which can be done for the full tree by tracing all references and each sub-array has a reference count equal to the number of seen references to it.
* Save an array and everything it references to a CAS.

### Generic Merkle Node

This is the mechanism for working with the generic Merkle DAG from DyND. A generic Merkle Node includes a set of named Merkle links and a set of node metadata. Its type is spelled `merkle_node`, and its data looks like

```
{
  node_type: string,
  links: map[string, { hash: bytes,
                       target_size: uint64,
                       target_type: string,
                       target_is_leaf: bool,
                       metadata: map[string, array[Any]]
                     }],
  metadata: map[string, array[Any]]
}
```

The `metadata` may not contain additional Merkle links via the Merkle Data Link type defined above.

How the `metadata` work, and how a generic Merkle node gets serialized to bytes for the CAS is part of the Merkle DAG structure, defined below. Since we don't have the `map` type defined/implemented in DyND yet, that's something which will be necessary.

### CAS Interface

Insertion of a DyND array into the CAS will work one of two ways. If the type of the array is `merkle_node`, its `node_type` is taken and passed to the CAS backend to use for additional metadata extraction. In other cases, the type "dynd" is passed to the backend together with the type of the array.

## Proposed Merkle DAG Structure

Entries in the CAS may be nodes or raw data of some kind. Nodes are always some kind of serialization of the generic Merkle node structure specified above.

Data in the CAS is not typed the way it is in git, so that the data there is exactly the data in question, not a transformed version of it. For example, this means a file `image.jpg` stored in the CAS contains exactly the file's contents, and its hash is exactly the hash of that file. The CAS subsystem can keep track of additional type metadata, by tracing through the Merkle DAG structure and grabbing the type information from outgoing links.

The two main ways raw data is encoded into the CAS is via "dynd" and "file" links. The rest of the Merkle DAG structure is mostly in terms of Merkle nodes which can be traversed and understood to a minimum extent by the general system.

### Merkle Link "dynd"

This is a link to data serialized according to the [DyND Serialization Design](serialization-design.md). The serialization and this link are designed with the aim of making the data as simple and obvious as possible, while being a unique serialization of particular data. The DyND type of the data is specified as part of the link metadata {"type": dyndtype} where the dyndtype value has DyND type `type`.

For example, an array of three integers [10, 20, 30] is the link with target_type "dynd", metadata {"type": ndt.type("3 * int32")}, and pointing at serialized bytes "0a 00 00 00 14 00 00 00 1e 00 00 00". Note that because of how DyND serialization was defined, these bytes are stored in an obvious "raw binary" format, identical to memory of C/C++ `int myarray[3];` on most modern platforms. This makes it possible to interoperate with a system that wants to share raw binary data but doesn't want to specifically use DyND.

### Merkle Node "node.filedir"

A "node.filedir" contains a directory of files, and consists fully of "file", "symlink", and "node.filedir" links.

#### Merkle Link "file"

A link to a file has `target_type` "file", `target_is_leaf` true, and possible metadata {"perms": unix_perms, "mimetype": mimetype}. The name of the link should be used as the filename. When linking directly to a file from some other type of node, the filename is instead placed in the metadata as {"name": filename}.

#### Merkle Link "symlink"

A link to a symlink has `target_type` "symlink", and is otherwise the same as for a file.

#### Merkle Link "node.directory"

A link to a directory tree has `target_type` "node.filedir", and is otherwise the same as for a file. When linking directly to a directory from some other type of node, the filename is instead placed in the metadata as {"name": dirname}.

### Merkle Node "node.doc.[`doctype`]"

A document node has node type "node.doc.[`doctype`]" where the `doctype` is a string representing the document type, for example the name of the application the document is for. The contents of the links and metadata are determined fully by the `doctype`. A general expectation of document nodes is that they contain relatively small data, with any bulk being linked to via "file" or "dynd" Merkle links.

A metadata entry "DCMI" should have type `map[string, array[Any]]`, and contain [Dublin Core Metadata](https://tools.ietf.org/html/rfc5013). That is pretty old, so maybe the more complicated DCMI abstract model described at http://dublincore.org/, using RDF, makes sense? Looks like it can get crazy pretty fast, and would add a lot of additional code dependencies.

#### Merkle Link to Earlier Version

Documents typically go through editing cycles, where parts of the document are changed and a new version of the document is created. Each time the document is saved, the document's graph is updated, with typically only a small subset of the nodes modified. Each save operation only needs to write the nodes which aren't in the CAS yet.

To track these changes, each document node includes a link to the previous version of itself. This is in the form of a link named "ancestor.*", pointing to another node of the same document type. One use for supporting multiple ancestor links is to distinguish between explicit user saves and autosaves. The link "ancestor.previous" might be to the most recent previous version, and "ancestor.usersave" might be to the most recent version which was explicitly saved by the user.

This way, it's possible to discard all the autosaved nodes, preserving just the user-saved history. An application could have a local temporary CAS, in which every single undoable operation is saved. Whenever the user says to save, the current document node and its full DAG also get saved into the CAS representing the document being worked on. Whenever the user says to publish, the current document node and its full DAG also get synchronized to some shared server.

## Merkle Node "node.docgraph.[`doctype`].[`docnodetype`]"

The document graph of a document is built out of nodes defined for that document type. This graph will consist of all the objects in the document, including user preferences, the UI state during save, etc.

## Merkle Node "node.expr.[`doctype`].[`exprnodetype`]"

Expressions will generally be a part of specific applications, for example [Dask](http://dask.pydata.org/en/latest/) might define a node like "node.expr.dask.dataframe.groupby". Each expression node type will define its input operands. It would be good to specify some conventions for this, so that generic code can have an understanding of the dependency graph of all expression DAGs.

Expression nodes will often be related to document graph nodes, but it is a good idea to think carefully about the difference. In general, the document graph node will be rich and full of incidental UI data, timestamps, authorship metadata, etc, while the expression node should be cut to the bare bones of parameters that actually affect the computation.

Consider an object represented within a 2D drawing program's document, like a clipping area. This might be the result of the union of two other clipping areas, so its clipping region is represented as a clipping region union expression node. For the user, there might be additional information saved about it, for example the UI might have a dropdown that chooses between union, intersection, etc, and each of those options might have additional parameters whose values should be saved between sessions for user comfort. Those additional values have no impact on the output of the union operation when a different clipping region operation is selected, so should not exist within the expression node.

By designing expression nodes so they contain only the information that affects the results of a computation, the dependencies are being expressed properly in the expression Merkle DAG, so an update/caching system can reliably use the hashes to determine when to update. This also maximizes the ability to reuse already computed values that have been saved into the CAS.

