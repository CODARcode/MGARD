#ifndef __SETTINGS_H__
#define __SETTINGS_H__

#include <string>

struct Settings {
    size_t L;
    int steps;
    int plotgap;
    double F;
    double k;
    double dt;
    double Du;
    double Dv;
    double noise;
    std::string output;
    bool checkpoint;
    int checkpoint_freq;
    std::string checkpoint_output;
    std::string adios_config;
    bool adios_span;
    bool adios_memory_selection;
    std::string mesh_type;

    Settings();
    static Settings from_json(const std::string &fname);
};

#endif
