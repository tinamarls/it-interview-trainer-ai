// fetchWithAuth.js
import { config } from "../config";

let logoutCallback = null; // глобальный колбэк на logout

export const setLogoutCallback = (callback) => {
    logoutCallback = callback;
};

export const fetchWithAuth = async (url, options = {}, token) => {
    const headers = {
        ...options.headers,
        Authorization: `Bearer ${token}`,
    };

    const response = await fetch(`${process.env.REACT_APP_API_URL}${url}`, {
        ...options,
        headers,
        credentials: "include",
        cache: "no-store",
    });

    if (response.status === 401) {
        if (logoutCallback) logoutCallback();
        throw new Error("Unauthorized");
    }

    return response;
};

