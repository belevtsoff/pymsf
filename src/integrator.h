#ifndef __PYX_HAVE__msf_tools
#define __PYX_HAVE__msf_tools


/* "msf_tools.pxd":11
 *     int end_idx
 * 
 * cdef public enum IntegrationType:             # <<<<<<<<<<<<<<
 *     EULER
 *     RK2
 */
enum IntegrationType {
  EULER,
  RK2,
  RK4
};

#ifndef __PYX_HAVE_API__msf_tools

#ifndef __PYX_EXTERN_C
  #ifdef __cplusplus
    #define __PYX_EXTERN_C extern "C"
  #else
    #define __PYX_EXTERN_C extern
  #endif
#endif

#endif /* !__PYX_HAVE_API__msf_tools */

#if PY_MAJOR_VERSION < 3
PyMODINIT_FUNC initmsf_tools(void);
#else
PyMODINIT_FUNC PyInit_msf_tools(void);
#endif

#endif /* !__PYX_HAVE__msf_tools */
