#pragma once

/**
 * kde_loader.hpp — Load precomputed KDE density vectors from CSV.
 * 
 * CSV format (no header): src_ip_int,dst_ip_int,kde_0,kde_1,...,kde_{dim-1}
 * 
 * Provides a lookup: (src_ip_str, dst_ip_str) → vector<double> of KDE dims.
 * IP addresses in the CSV are stored as unsigned 32-bit integers matching
 * the HyperVision .data format; they are converted to dotted-decimal strings
 * for lookup against long_edge src/dst strings.
 */

#include "../common.hpp"
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <vector>
#include <string>
#include <arpa/inet.h>

namespace Hypervision {

class kde_feature_loader {
private:
    // Key: "src_ip_str|dst_ip_str", Value: KDE density vector
    unordered_map<string, vector<double>> kde_map;
    size_t kde_dim = 0;
    bool loaded = false;

    static inline string uint32_to_ip_str(uint32_t ip) {
        struct in_addr addr;
        addr.s_addr = htonl(ip);
        return string(inet_ntoa(addr));
    }

    static inline string make_key(const string & src, const string & dst) {
        return src + "|" + dst;
    }

public:
    kde_feature_loader() = default;

    bool load_from_csv(const string & csv_path) {
        ifstream fin(csv_path);
        if (!fin.is_open()) {
            LOGF("KDE loader: cannot open %s", csv_path.c_str());
            return false;
        }

        kde_map.clear();
        string line;
        size_t count = 0;
        while (getline(fin, line)) {
            if (line.empty()) continue;
            
            // Parse: src_ip_int,dst_ip_int,v0,v1,...,v_{dim-1}
            stringstream ss(line);
            string token;
            vector<string> tokens;
            while (getline(ss, token, ',')) {
                tokens.push_back(token);
            }
            
            if (tokens.size() < 3) continue;
            
            // Map oversized IPs (e.g. IPv6) into uint32 range via modulo
            uint32_t src_int, dst_int;
            try {
                unsigned long val = stoul(tokens[0]);
                src_int = static_cast<uint32_t>(val);
            } catch (...) {
                // Number too large for stoul — hash the string into uint32
                std::hash<string> h;
                src_int = static_cast<uint32_t>(h(tokens[0]));
            }
            try {
                unsigned long val = stoul(tokens[1]);
                dst_int = static_cast<uint32_t>(val);
            } catch (...) {
                std::hash<string> h;
                dst_int = static_cast<uint32_t>(h(tokens[1]));
            }
            
            string src_str = uint32_to_ip_str(src_int);
            string dst_str = uint32_to_ip_str(dst_int);

            vector<double> vec;
            for (size_t i = 2; i < tokens.size(); i++) {
                vec.push_back(stod(tokens[i]));
            }

            if (kde_dim == 0) {
                kde_dim = vec.size();
            }

            kde_map[make_key(src_str, dst_str)] = std::move(vec);
            count++;
        }
        fin.close();
        loaded = true;
        LOGF("KDE loader: loaded %ld edge KDE vectors (dim=%ld) from %s", 
             count, kde_dim, csv_path.c_str());
        return true;
    }

    const vector<double> * lookup(const string & src_str, const string & dst_str) const {
        auto it = kde_map.find(make_key(src_str, dst_str));
        if (it != kde_map.end()) {
            return &(it->second);
        }
        return nullptr;
    }

    size_t get_kde_dim() const { return kde_dim; }
    bool is_loaded() const { return loaded; }
    size_t size() const { return kde_map.size(); }
};

} // namespace Hypervision
