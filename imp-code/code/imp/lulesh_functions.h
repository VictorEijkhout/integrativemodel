/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-6
 ****
 **** lulesh_functions.cxx : header file for lulesh support functions 
 ****
 ****************************************************************/

#ifndef LULESH_FUNCTIONS_H
#define LULESH_FUNCTIONS_H

#include <vector>

#include "imp_base.h"

/*
 * Element to local node
 */
domain_coordinate signature_coordinate_element_to_local( const domain_coordinate &i );
std::shared_ptr<multi_indexstruct> signature_struct_element_to_local
    (const multi_indexstruct&);
void element_to_local_function( kernel_function_types,int );

/*
 * Local node to global
 */
std::shared_ptr<multi_indexstruct> signature_local_from_global
    ( const multi_indexstruct &g,const multi_indexstruct &enc );
void local_to_global_function( kernel_function_types,const multi_indexstruct&);

/*
 * Global node to local
 */
std::shared_ptr<multi_indexstruct> signature_global_node_to_local( const multi_indexstruct &l );
void function_global_node_to_local( kernel_function_types,const multi_indexstruct& );

/*
 * Local node to element
 */
std::shared_ptr<multi_indexstruct> signature_local_to_element
    ( int dim,const multi_indexstruct &i );
void local_node_to_element_function( kernel_function_types );

#endif
