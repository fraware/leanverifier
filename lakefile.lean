import Lake
open Lake DSL

package formal_verif_ml {
}

require mathlib from git "https://github.com/leanprover-community/mathlib4.git"

lean_lib FormalVerifML {
  srcDir := "lean",  -- 🔍 Ensure this path matches your file structure!
  roots := #[`FormalVerifML]
}

@[default_target]
lean_exe formal_verif_ml_exe {
  root := `FormalVerifML.formal_verif_ml,  -- 🔍 Ensure this matches the module name!
  supportInterpreter := true,
  srcDir := "lean"
}
