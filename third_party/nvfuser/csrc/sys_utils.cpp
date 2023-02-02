#include <c10/util/Exception.h>

#if defined(__linux__)

#include <filesystem>
#include <string>
#include <vector>

#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <unistd.h>

namespace torch::jit::fuser::cuda::executor_utils {

std::string disassembleBinary(const std::vector<char>& cubin) {
  const char* err = "Failed to disassemble cubin";

  // Reference:
  // https://stackoverflow.com/a/3469651
  // https://linuxhint.com/dup2_system_call_c/
  pid_t pid;

  constexpr int READ = 0, WRITE = 1;
  int cubin_pipe[2];
  int disasm_pipe[2];

  TORCH_INTERNAL_ASSERT(pipe(cubin_pipe) == 0 && pipe(disasm_pipe) == 0, err);

  TORCH_INTERNAL_ASSERT((pid = fork()) != -1, err);

  if (pid) { // I am the parent
    // Parent only write cubin and read disasm, close unused pipe end
    TORCH_INTERNAL_ASSERT(close(cubin_pipe[READ]) == 0, err);
    TORCH_INTERNAL_ASSERT(close(disasm_pipe[WRITE]) == 0, err);

    // Wrap pipe fileno as C file stream
    FILE* cubin_fp = fdopen(cubin_pipe[WRITE], "wb");
    FILE* disasm_fp = fdopen(disasm_pipe[READ], "r");
    TORCH_INTERNAL_ASSERT(cubin_fp != nullptr, err);
    TORCH_INTERNAL_ASSERT(disasm_fp != nullptr, err);

    // Write cubin to nvdisasm
    size_t written = fwrite(cubin.data(), 1, cubin.size(), cubin_fp);
    TORCH_INTERNAL_ASSERT(written == cubin.size(), err);
    fclose(cubin_fp);

    // read disassembly result
    int ch;
    std::string result;
    result.reserve(cubin.size());
    while ((ch = fgetc(disasm_fp)) != EOF) {
      result.push_back(ch);
    }
    fclose(disasm_fp);
    return result;
  } else { // I am the child
    // For easier understanding, we can consider the fileno as a smart pointer
    // pointing to an underlying IO object in the kernel. Both the pointer and
    // the underlying objects are owned by the kernel, and multiple pointers
    // can point to the same object. `close` destroy the pointer, which does
    // not necessarily destroy the object.

    // Modify the stdin and stdout pointer to point to the pipe object
    TORCH_INTERNAL_ASSERT(close(STDIN_FILENO) == 0, err);
    TORCH_INTERNAL_ASSERT(close(STDOUT_FILENO) == 0, err);
    TORCH_INTERNAL_ASSERT(dup2(cubin_pipe[READ], STDIN_FILENO) != -1, err);
    TORCH_INTERNAL_ASSERT(dup2(disasm_pipe[WRITE], STDOUT_FILENO) != -1, err);

    // Now we have stdin and stdout pointing to the pipe object, we no longer
    // need the original pointers.
    TORCH_INTERNAL_ASSERT(close(cubin_pipe[READ]) == 0, err);
    TORCH_INTERNAL_ASSERT(close(cubin_pipe[WRITE]) == 0, err);
    TORCH_INTERNAL_ASSERT(close(disasm_pipe[READ]) == 0, err);
    TORCH_INTERNAL_ASSERT(close(disasm_pipe[WRITE]) == 0, err);

    // If execl succeed, then the current process will be replaced by nvdisasm,
    // and all the remaining code in the current process will not be executed.
    // So, execl only returns when it fails.
    //
    // TODO: I was planning to use `nvdisasm /dev/stdin` which could avoid
    // creating temporary file, but unfortunately, that fails with:
    //   nvdisasm fatal   : Memory allocation failure
    // so I have to dump the stdin to a temp file and let nvdisasm read it. I am
    // hoping that nvdisasm will support reading from stdin one day.
    execl(
        "/bin/bash",
        "bash",
        "-c",
        "TMPFILE=$(mktemp); cat>$TMPFILE; nvdisasm $TMPFILE; rm $TMPFILE",
        NULL);

    // only reachable when execl fails
    TORCH_INTERNAL_ASSERT(false, err);
  }
}

} // namespace torch::jit::fuser::cuda::executor_utils

#else

namespace torch::jit::fuser::cuda::executor_utils {

std::string disassembleBinary(const std::vector<char>& binary) {
  TORCH_CHECK(false, "disassembling cubin is only supported on Linux");
}

} // namespace torch::jit::fuser::cuda::executor_utils

#endif
