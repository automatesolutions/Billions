import { auth } from "@/auth";
import { NextResponse } from "next/server";

export default auth((req) => {
  const { nextUrl } = req;
  const isLoggedIn = !!req.auth;

  // Define protected routes
  const isProtectedRoute = nextUrl.pathname.startsWith("/dashboard") ||
    nextUrl.pathname.startsWith("/analyze") ||
    nextUrl.pathname.startsWith("/outliers") ||
    nextUrl.pathname.startsWith("/portfolio");

  // Redirect to login if accessing protected route while not logged in
  if (isProtectedRoute && !isLoggedIn) {
    return NextResponse.redirect(new URL("/login", nextUrl));
  }

  // Redirect to dashboard if accessing login while already logged in
  if (nextUrl.pathname === "/login" && isLoggedIn) {
    return NextResponse.redirect(new URL("/dashboard", nextUrl));
  }

  return NextResponse.next();
});

// Configure which routes use this middleware
export const config = {
  matcher: [
    /*
     * Match all request paths except for the ones starting with:
     * - api (API routes)
     * - _next/static (static files)
     * - _next/image (image optimization files)
     * - favicon.ico (favicon file)
     * - public files (public folder)
     */
    "/((?!api|_next/static|_next/image|favicon.ico|.*\\.png$|.*\\.jpg$|.*\\.svg$).*)",
  ],
};

