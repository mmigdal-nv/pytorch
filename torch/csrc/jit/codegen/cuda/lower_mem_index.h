#pragma once
#include <c10/macros/Export.h>

#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir.h>

#include <vector>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

struct AddressRecordKey;

class AddressRecord {
 public:
  //! Utility class to note the read or write
  //!  direction of this address.
  enum class ReadWrite { READ, WRITE };

  explicit AddressRecord(
      TensorView* data_tv,
      TensorView* address_tv,
      std::vector<IterDomain*> allocation_ids,
      TensorView* reference_tv,
      ReadWrite direction,
      IterDomain* serial_id);

  bool isRead() const {
    return access_direction_ == ReadWrite::READ;
  }

  bool isWrite() const {
    return access_direction_ == ReadWrite::WRITE;
  }

  TensorView* dataTensor() const {
    return data_tv_;
  }

  TensorView* addressTensor() const {
    return address_tv_;
  }

  TensorView* indexReferenceTensor() const {
    return reference_tv_;
  }

  const auto& allocationIterDomains() const {
    return allocation_ids_;
  }

  AddressRecordKey key() const;

  IterDomain* getConcreteSerialLoopId() const {
    return loop_concrete_serial_id_;
  }

 private:
  //! The address tensor that will hold the
  //!  data address to the access_tv.
  TensorView* address_tv_;

  //! The tensorview that this address record
  //!  will save address for.
  TensorView* data_tv_;

  //! The tensorview that will be the consumer
  //!  if this is a record for the read access,
  //!  and would be ignored for write access since
  //!  the access tv would have all the info.
  TensorView* reference_tv_;

  //! Records if this is a read adddress or a write
  //!  address.
  ReadWrite access_direction_ = ReadWrite::WRITE;

  //! Loop id's that correspond to the allocation iterdomain
  //!  of this
  std::vector<IterDomain*> allocation_ids_;

  //! Loop id that this address record will be lifted
  //!  out of.
  IterDomain* loop_concrete_serial_id_;
};

// Utility class to index address record
struct AddressRecordKey {
  const TensorView* reference_tv = nullptr;
  const TensorView* data_tv = nullptr;

  AddressRecordKey(TensorView* reference_tv_, TensorView* data_tv_)
      : reference_tv(reference_tv_), data_tv(data_tv_) {}
};

struct AddressRecordKeyHash {
  std::size_t operator()(const AddressRecordKey& key) const {
    auto h1 = std::hash<const TensorView*>{}(key.reference_tv);
    auto h2 = std::hash<const TensorView*>{}(key.data_tv);
    return h1 ^ h2;
  }
};

class AddressComputeInfo {
 public:
  void build(Fusion* fusion);

  c10::optional<AddressRecord*> getMaybeLiftedAddress(
      TensorView* data_tv,
      TensorView* reference_tv = nullptr);

  c10::optional<AddressRecord*> getMaybeRecordForAddressTv(TensorView* tv);

 private:
  // Utility to help allocate space for saving pre-computed address.
  TensorView* makeAddressTv(
      std::vector<IterDomain*> address_domains,
      bool is_global_address);

  void makeAddressRecord(TensorView* data_tv, TensorView* reference_tv);

 private:
  using AddressRecordPtr = std::unique_ptr<AddressRecord>;

  // Collected records of all the indexing math that needs
  //  to be lifted, indexed with reference tv and data_tv.
  std::unordered_map<AddressRecordKey, AddressRecordPtr, AddressRecordKeyHash>
      index_lift_record_;

  //! Short cut from the address tensorview to
  //!  the address record information.
  std::unordered_map<TensorView*, AddressRecord*> address_tv_to_address_record_;
};

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
