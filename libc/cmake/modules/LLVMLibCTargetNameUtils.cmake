function(get_fq_target_name local_name target_name_var)
  file(RELATIVE_PATH rel_path ${LIBC_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
  string(REPLACE "/" "." fq_name "libc.${rel_path}.${local_name}")
  set(${target_name_var} ${fq_name} PARENT_SCOPE)
endfunction(get_fq_target_name)

function(is_relative_target_name target_name output_var)
  string(FIND ${target_name} "." dot_loc)
  string(COMPARE EQUAL "0" ${dot_loc} is_relative)
  set(${output_var} ${is_relative} PARENT_SCOPE)
endfunction(is_relative_target_name)

function(get_fq_dep_name fq_name name)
  is_relative_target_name(${name} "is_relative")
  if(is_relative)
    # Skip over the first '.' character.
    string(SUBSTRING ${name} 1 -1 local_name)
    get_fq_target_name(${local_name} fully_qualified_name)
    set(${fq_name} ${fully_qualified_name} PARENT_SCOPE)
  else()
    set(${fq_name} ${name} PARENT_SCOPE)
  endif()
endfunction(get_fq_dep_name)

function(get_fq_deps_list output_list)
  set(fq_dep_name_list "")
  foreach(dep IN LISTS ARGN)
    get_fq_dep_name(fq_dep_name ${dep})
    list(APPEND fq_dep_name_list ${fq_dep_name})
  endforeach(dep)
  set(${output_list} ${fq_dep_name_list} PARENT_SCOPE)
endfunction(get_fq_deps_list)

# Target types
set(COMMAND_TARGET_TYPE "COMMAND")
set(HEADER_TARGET_TYPE "HEADER")
set(HDR_LIBRARY_TARGET_TYPE "HDR_LIBRARY")
set(OBJECT_LIBRARY_TARGET_TYPE "OBJECT_LIBRARY")
set(ENTRYPOINT_EXT_TARGET_TYPE "ENTRYPOINT_EXT")
set(ENTRYPOINT_OBJ_TARGET_TYPE "ENTRYPOINT_OBJ")
set(ENTRYPOINT_OBJ_VENDOR_TARGET_TYPE "ENTRYPOINT_OBJ_VENDOR")
set(ENTRYPOINT_LIBRARY_TARGET_TYPE "ENTRYPOINT_LIBRARY")

set(DEP_PROPAGATE_TARGET_TYPE
    HEADER_TARGET_TYPE
    HDR_LIBRARY_TARGET_TYPE
    OBJECT_LIBRARY_TARGET_TYPE
    ENTRYPOINT_EXT_TARGET_TYPE
    ENTRYPOINT_OBJ_TARGET_TYPE
    ENTRYPOINT_OBJ_VENDOR_TARGET_TYPE
)

# TODO: Make check_alias works for multiple aliasing levels.
function(check_alias output target_name)
  if(NOT TARGET ${target_name})
    message(FATAL_ERROR "Target ${target_name} does not exist.")
  endif()

  get_target_property(alias ${target_name} ALIASED_TARGET)
  if(alias)
    set(${output} ${alias} PARENT_SCOPE)
  else()
    set(${output} ${target_name} PARENT_SCOPE)
  endif()
endfunction(check_alias)

function(get_linkable_targets output_list)
  set(all_deps "")
  set(${output_list} ${all_deps} PARENT_SCOPE)
  foreach(target_name IN LISTS ARGN)
    if(NOT TARGET ${target_name})
      continue()
    endif()

    check_alias(target ${target_name})
    get_target_property(target_type ${target} TARGET_TYPE)
    if(${target_type} IN_LIST DEP_PROPAGATE_TARGET_TYPE)
      list(APPEND all_deps ${target})
    endif()
  endforeach()

  set(${output_list} ${all_deps} PARENT_SCOPE)
endfunction()

# Propagate all dependencies of certain target types.
function(get_all_target_deps output_list)
  set(all_deps "")
  foreach(dep IN LISTS ARGN)
    if(NOT TARGET ${dep})
      continue()
    endif()
    check_alias(target ${dep})
    get_target_property(target_deps ${target} DEPS)
    get_linkable_targets(target_objs ${target_deps})
    list(APPEND all_deps ${target_objs})
  endforeach()
  set(${output_list} ${all_deps} PARENT_SCOPE)
endfunction(get_all_target_deps)

function(get_all_deps output_list)
  set(all_deps ${ARGN})
  get_all_target_deps(deps ${ARGN})
  list(APPEND all_deps ${deps})
  list(REMOVE_DUPLICATES all_deps)
  set(${output_list} ${all_deps} PARENT_SCOPE)
endfunction(get_all_deps)