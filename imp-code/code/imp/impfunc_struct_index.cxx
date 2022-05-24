/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-9
 ****
 **** impfunc_struct_index : include file with common structure definitions
 ****
 ****************************************************************/

data_pointer outdata;
try {
  outdata = outvector->get_data(p);
 } catch (std::string c) {
  throw(fmt::format("Could not get outdata for p={}: {}",p.as_string(),c));
 } catch (...) {
  throw(fmt::format("Could not get outdata for p={}",p.as_string()));
 }

std::vector<index_int> pfirst,plast,offsets,nsize;
try {
  pfirst = outdistro->first_index_r(p).data();
  plast  = outdistro->last_index_r(p).data();
  offsets = outdistro->offset_vector().data();
  nsize = outdistro->get_numa_structure()->local_size_r().data();
 } catch (std::out_of_range) {
  throw(fmt::format("Could not get int vectors because out of range, p={}",
                    p.as_string()));
 } catch (...) {
  throw(fmt::format("Could not get int vectors, p={}",p.as_string()));
 }

