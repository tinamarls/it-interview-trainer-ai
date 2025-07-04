// // AuthProvider.js
// import React, { createContext, useState, useEffect } from "react";
// import { setLogoutCallback } from "../../api/fetchWithAuth";
//
// export const AuthContext = createContext();
//
// export const AuthProvider = ({ children }) => {
//     const [token, setToken] = useState(null);
//
//     useEffect(() => {
//         const savedToken = localStorage.getItem("token");
//         if (savedToken) {
//             setToken(savedToken);
//         }
//     }, []);
//
//     const logout = (onLogout) => {
//         localStorage.removeItem("token");
//         setToken(null);
//         if (typeof onLogout === 'function') {
//             onLogout();
//         }
//     };
//
//     useEffect(() => {
//         setLogoutCallback(() => logout());
//     }, []);
//
//     return (
//         <AuthContext.Provider value={{ token, setToken, logout }}>
//             {children}
//         </AuthContext.Provider>
//     );
// };
import React, { createContext, useState, useEffect } from "react";

export const AuthContext = createContext();

export const AuthProvider = ({ children }) => {
    const [token, setToken] = useState(() => {
        return localStorage.getItem("authToken") || null;
    });
    const [user, setUser] = useState(null);
    const [isAuthenticated, setIsAuthenticated] = useState(false);
    const [isLoading, setIsLoading] = useState(true);

    // Эффект для проверки авторизации при загрузке
    useEffect(() => {
        const storedToken = localStorage.getItem("authToken");
        if (storedToken) {
            setToken(storedToken);
            setIsAuthenticated(true);
            // Можно добавить проверку валидности токена здесь
            // fetchUserData(storedToken);
        }
        setIsLoading(false); // Важно: устанавливаем isLoading в false после инициализации
    }, []);

    const handleLogin = (newToken) => {
        try {
            localStorage.setItem("authToken", newToken);
            setToken(newToken);
            setIsAuthenticated(true);
        } catch (error) {
            console.error("Ошибка при входе:", error);
        }
    };

    const handleLogout = () => {
        localStorage.removeItem("authToken");
        setToken(null);
        setUser(null);
        setIsAuthenticated(false);
    };

    return (
        <AuthContext.Provider
            value={{
                token,
                user,
                isAuthenticated,
                isLoading,
                login: handleLogin,
                logout: handleLogout
            }}
        >
            {children}
        </AuthContext.Provider>
    );
};