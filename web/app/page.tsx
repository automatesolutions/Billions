import Image from "next/image";
import Link from "next/link";
import { auth } from "@/auth";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

export default async function Home() {
  const session = await auth();

  return (
    <div className="min-h-screen bg-black relative">
      <div className="absolute inset-0 bg-gradient-radial from-transparent via-yellow-900/20 to-transparent"></div>
      <div className="max-w-4xl mx-auto p-8 space-y-8 relative z-10">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Image
              src="/logo1.png"
              alt="BILLIONS Logo"
              width={64}
              height={64}
              priority
            />
            <div>
              <h1 className="text-4xl font-black text-transparent bg-clip-text bg-gradient-to-b from-yellow-200 via-yellow-400 to-yellow-600" style={{
                textShadow: '0 1px 0 rgba(255, 255, 255, 0.8), 0 0 20px rgba(255, 215, 0, 0.6), 0 0 40px rgba(255, 215, 0, 0.4)',
                filter: 'drop-shadow(0 0 10px rgba(255, 215, 0, 0.8))'
              }}>
                BILLIONS
              </h1>
              <p className="text-sm font-medium text-white" style={{
                textShadow: '0 0 10px rgba(255, 255, 255, 0.3), 0 0 20px rgba(255, 255, 255, 0.2)',
                filter: 'drop-shadow(0 0 5px rgba(255, 255, 255, 0.4))'
              }}>
                Quant trading made easy.
              </p>
            </div>
          </div>
          
          {/* Red Box - Log In and Sign In */}
          <div className="flex gap-2">
            <Link href="/login">
              <Button className="bg-gray-800 hover:bg-gray-700 text-white border-gray-600">
                Log In
              </Button>
            </Link>
            <Link href="/login">
              <Button className="bg-gray-800 hover:bg-gray-700 text-white border-gray-600">
                Sign In
              </Button>
            </Link>
          </div>
        </div>

        {/* Yellow Box - The Mindset Matrix */}
        <div className="text-center py-4">
          <h2 className="text-2xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-yellow-300 via-yellow-400 to-yellow-600" style={{
            textShadow: '0 0 20px rgba(255, 215, 0, 0.6), 0 0 40px rgba(255, 215, 0, 0.4)',
            filter: 'drop-shadow(0 0 10px rgba(255, 215, 0, 0.6))'
          }}>
            The Mindset Matrix: Where Perception Generates Prosperity
          </h2>
        </div>

        {/* Green Box - Video */}
        <div className="w-full">
          <video 
            className="w-full h-auto rounded-lg"
            controls
            autoPlay
            muted
            loop
          >
            <source src="/home_video.mp4" type="video/mp4" />
            Your browser does not support the video tag.
          </video>
        </div>
      </div>
    </div>
  );
}
