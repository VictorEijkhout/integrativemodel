std::shared_ptr<distribution> indistro,
  outdistro = outvector->get_distribution();
int
  dim = p.get_same_dimensionality(outdistro->get_dimensionality()),
  k   = outdistro->get_orthogonal_dimension();

data_pointer indata,outdata;
try {
  outdata = outvector->get_data(p);
 } catch (std::string c) {
  throw(fmt::format("Could not get outdata for p={}: {}",p.as_string(),c));
 } catch (...) {
  throw(fmt::format("Could not get outdata for p={}",p.as_string()));
 }

//std::vector<index_int> pfirst,plast,offsets,nsize;
domain_coordinate pfirst,plast,qfirst,qlast,
  in_offsets,in_nsize, out_offsets,out_nsize;
try {
  pfirst = outdistro->first_index_r(p);
  plast  = outdistro->last_index_r(p);
  out_offsets = outdistro->offset_vector();
  out_nsize   = outdistro->get_numa_structure()->local_size_r();
 } catch (std::out_of_range) {
  throw(fmt::format("Could not get int vectors because out of range, p={}",
                    p.as_string()));
 } catch (...) {
  throw(fmt::format("Could not get int vectors, p={}",p.as_string()));
 }

std::shared_ptr<object> invector;
if (invectors.size()>0) {
  invector   = invectors.at(0);
  indistro   = invector->get_distribution();
  qfirst     = indistro->first_index_r(p);
  qlast      = indistro->last_index_r(p);
  indata     = invector->get_data(p);
  in_offsets = indistro->offset_vector();
  in_nsize   = indistro->get_numa_structure()->local_size_r();
}
