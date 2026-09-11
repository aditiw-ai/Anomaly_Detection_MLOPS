import axios from "axios";
import { config } from "../config";

// Debug (remove later)
console.log("ENV:", config.ENV);
console.log("API BASE URL:", config.API_BASE_URL);

export const api = axios.create({
    baseURL: config.API_BASE_URL,
    timeout: 15000,
    headers: {
        "Content-Type": "application/json",
    },
});

// 🔐 Attach token automatically
api.interceptors.request.use(
    (config) => {
        const token = localStorage.getItem("token") || localStorage.getItem("access_token");

        if (token) {
            config.headers.Authorization = `Bearer ${token}`;
        }

        return config;
    },
    (error) => Promise.reject(error)
);

// 🚨 Global error handling
api.interceptors.response.use(
    (response) => response,
    (error) => {
        if (error.response?.status === 401) {
            console.error("Unauthorized");

            localStorage.removeItem("token");
            localStorage.removeItem("access_token");
            localStorage.removeItem("user");
            window.dispatchEvent(new Event("shadow-hubble:auth-expired"));
            // window.location.href = "/login"; (optional)
        }

        return Promise.reject(error);
    }
);
