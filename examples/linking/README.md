# Linking Against MGARD

First, build and install MGARD.
Then, run the following in `examples/linking`.

```console
$ cmake -S . -B build
$ cmake --build build
```

If `build/main` is created, you have successfully linked against MGARD.

```console
$ build/main
you have successfully linked against MGARD
```

See `CMakeLists.txt` for some hints if you encounter problems finding dependencies.
