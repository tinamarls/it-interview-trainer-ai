import React, { useContext } from "react";
import { Navigate } from "react-router-dom";
import { AuthContext } from "../AuthProvider/AuthProvider";

// export const ProtectedRoute = ({ children }) => {
//     const { token } = useContext(AuthContext);
//
//     if (!token) {
//         return <Navigate to="/sign-in" />;
//     }
//
//     return children;
// };

export const ProtectedRoute = ({ children }) => {
    const { token, isLoading } = useContext(AuthContext);

    if (isLoading) {
        return <div>Загрузка...</div>; // можно спиннер
    }

    if (!token) {
        return <Navigate to="/sign-in" />;
    }

    return children;
};
