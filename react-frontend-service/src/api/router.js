import * as React from "react";
import { createBrowserRouter } from "react-router-dom";
import { SignIn } from "../pages/SignIn/SignIn.jsx";
import { SignUp } from "../pages/SignUp/SignUp.jsx";
import { Resume } from "../pages/Resume/Resume.jsx";
import { StressTrain } from "../pages/StressTrain/StressTrain.jsx";
import { Header } from "../components/Header";
import { ProtectedRoute } from "../pages/AuthProvider/ProtectedRoute.jsx";
import { Outlet } from "react-router-dom";
import {Vacancy} from "../pages/Vacancy/Vacancy";
import {Compare} from "../pages/Compare/Compare";
import InterviewSimulator from "../pages/InterviewSimulator/InterviewSimulator";
import HrInterviewResults from "../pages/HrInterviewResults/HrInterviewResults";
import HrInterviewResultDetails from "../pages/HrInterviewResults/HrInterviewResultDetails";

// Обёртка с Header
const Root = () => (
    <>
        <Header />
        <main style={{ padding: "20px" }}>
            <Outlet />
        </main>
    </>
);

// Маршруты
export const router = createBrowserRouter([
    {
        path: "/",
        element: <Root />,
        children: [
            {
                path: "resume",
                element: (
                    <ProtectedRoute>
                        <Resume />
                    </ProtectedRoute>
                ),
            },
            {
                path: "interview",
                element: (
                    <ProtectedRoute>
                        <StressTrain />
                    </ProtectedRoute>
                ),
            },
            {
                path: "vacancy",
                element: (
                    <ProtectedRoute>
                        <Vacancy />
                    </ProtectedRoute>
                ),
            },
            {
                path: "compare",
                element: (
                    <ProtectedRoute>
                        <Compare />
                    </ProtectedRoute>
                ),
            },
            {
                path: "hr-interview",
                element: (
                    <ProtectedRoute>
                        <InterviewSimulator />
                    </ProtectedRoute>
                ),
            },
            {
                path: "hr-interview/results",
                element: (
                    <ProtectedRoute>
                        <HrInterviewResults />
                    </ProtectedRoute>
                ),
            },
            {
                path: "hr-interview/results/:attemptId",
                element: (
                    <ProtectedRoute>
                        <HrInterviewResultDetails />
                    </ProtectedRoute>
                ),
            },
        ],
    },
    { path: "/sign-in", element: <SignIn /> },
    { path: "/sign-up", element: <SignUp /> },
]);
