Debugging with the VIXL Simulator
=================================

The VIXL AArch64 simulator contains a basic debugger which can be used to debug
simulated applications. The debugger supports basic debugging features such as
setting breakpoints, stepping through simulated instructions and printing
simulator specific information, for example: printing the values of a register
or printing instructions at specified addresses.

Using the Debugger
------------------

In order to use the debugger it first needs to be enabled in the simulator.

```C++
    Decoder decoder;
    Simulator simulator(&decoder);
    simulator.SetDebuggerEnabled(true);
```

Once enabled, the debugger will be activated whenever a breakpoint (brk) is
encountered by the simulator. For example:

```asm
    add x1, x0, #5
    mov x2, #2

    brk 0   // Debugger activated here.

    sub x3, x1, x2
```

Further breakpoints can be set either programmatically or interactively in the
debugger itself. For example, to set breakpoints programmatically:

```C++
    // 'func' is an AARCH64 assembly function.
    extern "C" void func();

    Debugger* debugger = simulator.GetDebugger();

    // Register a breakpoint at a fixed (absolute) address.
    debugger->RegisterBreakpoint(0x00007ffbc6d38000);

    // Register a breakpoint to an already existing assembly function.
    debugger->RegisterBreakpoint(reinterpret_cast<uint64_t>(&func));
```

Or to set breakpoints interactively once the debugger has been activated:

```sh
    sim> break 0x00007ffbc6d38000
```

The debugger has a variety of useful commands to control program flow (e.g:
step, next, continue) and inspect features of the running simulator (e.g:
print, trace). To view a list of all supported commands
use "help" at the debugger prompt.

```sh
    sim> help
```

Extending the Debugger
----------------------

The debugger can be extended with custom commands to allow for greater
flexibility in debugging individual applications. This could be used for a
variety of applications, for example printing out object specific information
from an address.

To create a custom debugger command, extend the DebuggerCmd class located in
debugger-aarch64.h and implement its methods.

```C++
    class PrintObjectCmd : public DebuggerCmd {
     public:
      PrintObjectCmd(Simulator* sim)
            : DebuggerCmd(sim,
                          "printobject",
                          "po",
                          "<address>",
                          "Print a custom object located at the given address.")
      {}

      // Called when the command word is given to the interactive debugger.
      DebugReturn Action(const std::vector<std::string>& args) override {
        // We want exactly 1 argument (an address) given to the printobject
        // command.
        if (args.size() != 1) {
            fprintf(ostream_, "Error: incorrect command format.");
            return DebugContinue;
        }

        auto addr = Debugger::ParseUint64String(args.front());
        if (addr) {
            fprintf(ostream_, "Error: could not get address from string.");
            return DebugContinue;
        }

        // Convert the address given to a custom object and then print it.
        CustomObject object = reinterpret_cast<CustomObject>(*addr);
        object.print();
      }
    };
```

Then simply register the new command with the debugger.

```C++
    Debugger* debugger = simulator.GetDebugger();

    debugger->RegisterCmd<PrintObjectCmd>();
```
