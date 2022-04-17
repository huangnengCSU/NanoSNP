#ifndef __BED_INTV_LIST_HPP
#define __BED_INTV_LIST_HPP

#include "cpp_aux.hpp"
#include "ref_reader.hpp"

#include <map>
#include <string>

class BedIntvList
{
public:
    BedIntvList(reference_struct* reference, const char* bed_path);
    ~BedIntvList();

    bool region_intersect_with_bed_intv(int oid, int from, int to) const;
    bool region_intersect_with_bed_intv(const std::string& name, int from, int to) const;

private:
    reference_struct* m_reference;
    std::vector<std::pair<int, int>> m_intv_list;
    std::vector<char> m_intv_endpoint_list;
};

#endif // __BED_INTV_LIST_HPP