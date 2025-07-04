import React from "react";
import { RouterProvider } from "react-router-dom";
import { router } from "./api/router";
import 'bootstrap/dist/css/bootstrap.min.css';

function App() {
  return <RouterProvider router={router} />;
}

export default App;
